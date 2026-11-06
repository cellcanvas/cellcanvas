"""Model providers a service instance can use, and what each one can actually do.

CellCanvas separates three kinds of model capability, because no model offers all of them:

- **features**: a frozen model that produced per-voxel features on the reconstruction grid. The service reads saved
  features with their manifest; it does not run feature extraction itself (``scripts/extract_dinov2_features.py``
  is the public DINOv2 producer).
- **classifiers**: an interactive head trained on painted voxels over those features (``feature_learning.train`` and
  ``feature_learning.predict``).
- **prompt backends**: promptable segmentation (points, boxes, scribbles) served by another backend, such as an
  nnInteractive server. The service only lists their URLs, so clients can call them directly; it does not proxy
  or imitate prompting.

Private providers (for example an internal foundation-model adapter) are installed as separate packages that
register a ``cellcanvas.providers`` entry point. They are loaded only when a deployment profile names them and
allows private providers; their features are never reported as public.
"""

from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError, entry_points, version

SEEDING_POLICY = "inferred_background"
"""Training label policy that adds inferred background rows from unknown voxels (cellcanvas-core#20)."""

SEEDING_IMPLEMENTATION = "cellcanvas_core.data.seeding.SeededDataManager"

SEEDING_PARAMETERS = {"percentile": float, "max_seeds": int, "n_candidates": int, "random_state": int}
"""SeededDataManager parameters a client may set. background_label always comes from the dataset's labels."""

ENTRY_POINT_GROUP = "cellcanvas.providers"
MANIFEST_FIELDS = ("model", "license", "public", "source", "grid")


def without_local_paths(value):
    """A copy of a JSON value without strings that are local filesystem paths (absolute or file:// URLs)."""
    if isinstance(value, dict):
        kept = {key: without_local_paths(inner) for key, inner in value.items()}
        return {key: inner for key, inner in kept.items() if inner is not None}
    if isinstance(value, list):
        return [without_local_paths(inner) for inner in value]
    if isinstance(value, str) and (value.startswith(("/", "file:", "~")) or value[1:3] in (":\\", ":/")):
        return None
    return value


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def seeding_available():
    """Whether the installed cellcanvas-core provides SeededDataManager (absent before cellcanvas-core#20)."""
    try:
        import_module("cellcanvas_core.data.seeding")
    except ImportError:
        return False
    return True


def random_forest():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(random_state=0, n_jobs=1)


def fewshot():
    from cellcanvas_core.semantic.fewshot import FewShotClassifier

    # DataManager already drops unknown (0) voxels and shifts classes to start at 0.
    return FewShotClassifier(unlabelled=None)


def fewshot_available():
    try:
        import_module("cellcanvas_core.semantic.fewshot")
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class Classifier:
    name: str
    factory: object
    description: str
    available: object = staticmethod(lambda: True)


CLASSIFIERS = {
    "rf": Classifier("rf", random_forest,
                                "scikit-learn random forest (random_state=0) on the feature rows"),
    "fewshot": Classifier("fewshot", fewshot, "cellcanvas-core FewShotClassifier (k-NN bank and MLP head)",
                          fewshot_available),
}


@dataclass
class FeatureSource:
    """Saved per-voxel features of one dataset and their provenance manifest."""

    manifest: dict
    provider: str = "saved"

    def __post_init__(self):
        missing = [key for key in MANIFEST_FIELDS if key not in self.manifest]
        if missing:
            raise ValueError(f"Features manifest lacks {missing}")
        if not isinstance(self.manifest["public"], bool):
            raise ValueError("Features manifest 'public' must be true or false")

    @property
    def public(self):
        return self.manifest["public"]

    def model_info(self):
        return {"name": self.manifest["model"], "public": self.public, "license": self.manifest["license"]}

    def provenance(self):
        """Feature provenance that can leave this machine: the manifest without local paths (checkpoint and source
        digests are kept)."""
        return without_local_paths(self.manifest)


@dataclass
class Registry:
    """Classifiers, private feature providers and prompt backends available to one service instance."""

    classifiers: dict = field(default_factory=lambda: dict(CLASSIFIERS))
    feature_providers: dict = field(default_factory=dict)
    prompt_backends: list = field(default_factory=list)
    allow_private: bool = False

    def classifier(self, name):
        entry = self.classifiers.get(name)
        if entry is None or not entry.available():
            raise KeyError(name)
        return entry.factory()

    def describe(self):
        return {
            "classifiers": [{"name": c.name, "description": c.description} for c in self.classifiers.values()
                            if c.available()],
            "label_policies": [SEEDING_POLICY] if seeding_available() else [],
            "feature_providers": [{"name": name, "public": getattr(provider, "public", False)}
                                  for name, provider in self.feature_providers.items()],
            "prompt_backends": [dict(backend) for backend in self.prompt_backends],
            "allow_private": self.allow_private,
        }

    def check_features(self, source):
        """Refuse private features unless this deployment allows a registered private provider for them."""
        if source.public:
            return
        if not self.allow_private:
            raise ValueError("Features are not from a public model; this deployment does not allow private providers")
        if source.provider not in self.feature_providers:
            raise ValueError(f"Private features name provider {source.provider!r}, which is not registered")
        self.feature_providers[source.provider].check_manifest(source.manifest)

    def load_plugins(self, names):
        """Load named ``cellcanvas.providers`` entry points; each returns a provider with ``name``, ``public``
        and ``check_manifest(manifest)``."""
        available = {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}
        for name in names:
            if name not in available:
                raise ValueError(f"Provider plugin {name!r} is not installed (group {ENTRY_POINT_GROUP})")
            provider = available[name].load()()
            if not provider.public and not self.allow_private:
                raise ValueError(f"Provider {name!r} is private; set providers.allow_private in the profile")
            self.feature_providers[provider.name] = provider
