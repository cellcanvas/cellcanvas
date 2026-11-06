"""Serve registered core datasets without viewer or Copick dependencies.

Operations run in worker threads and serialize per dataset. A disconnected
request does not cancel core execution. Prediction arrays are persisted by
core, whose writes can be partial after a failure. With a state directory
(see ``state``), receipts, trained models and published-result manifests
survive a restart; without one they last for this service instance.

Training can add inferred background rows through the core seeding policy;
they are recorded as provenance next to the model and never written into the
painting, where 0 stays unknown.

Painting can be replaced over HTTP under the dataset lock, so an external
annotation store (for example a copick segmentation) can be synchronized
before training. A painting upload may state the voxel grid it was made on;
a grid that differs from the dataset's is refused. A dataset may declare its
label classes (0 is always unknown); painting with undeclared values is refused.

A prediction is readable only after it completed, by its version, with the
digest of the stored voxels, so clients never publish a partial write.
Prediction can be limited to a region (for example one slice of a
full-resolution volume); that version then consists of the region only, and
reading it returns the region's voxels with their bounds.

Features may be stored per voxel or as a patch-token grid that core
interpolates on read (``cellcanvas_patch_grid`` metadata on the Zarr array);
the config is the same, ``features_path`` names either. Each
completed prediction has a result record: the painting revision and digest it
was trained on, the classifier, inferred seeds, feature provenance and labels.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock, Thread
from uuid import UUID

import numpy as np
from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager
from sklearn.ensemble import RandomForestClassifier
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .providers import (SEEDING_IMPLEMENTATION, SEEDING_PARAMETERS, SEEDING_POLICY, FeatureSource, Registry,
                        package_version, seeding_available)
from .state import MemoryStore

DEFAULT = object()
UNIT_ANGSTROM = {"angstrom": 1.0, "nanometer": 10.0, "micrometer": 1e4}


def label_digest(array):
    """sha256 of dtype, shape and C-order bytes, so equal digests mean equal label arrays."""
    array = np.ascontiguousarray(array)
    header = f"{array.dtype.str}:{','.join(map(str, array.shape))}:".encode()
    return "sha256:" + hashlib.sha256(header + array.tobytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def backend_version():
    return f"cellcanvas {package_version('cellcanvas')}; cellcanvas-core {package_version('cellcanvas-core')}"


@dataclass(frozen=True)
class Labels:
    """What painted values mean. 0 is always unknown and never trained on; background is a class people paint.

    Config: ``{"classes": {"1": "background", "2": "membrane"}, "background": 1}``. ``background`` names the
    class that inferred background seeds train as; without it, seeding is not offered.
    """

    classes: dict
    background: int | None = None

    @classmethod
    def from_config(cls, config):
        if not isinstance(config, dict) or set(config) - {"classes", "background"} or not config.get("classes"):
            raise ValueError("labels must be {classes: {value: name}, background: value or null}")
        classes = {int(value): str(name) for value, name in config["classes"].items()}
        if min(classes) < 1:
            raise ValueError("Label 0 means unknown; classes must be positive values")
        background = config.get("background")
        if background is not None and background not in classes:
            raise ValueError(f"background {background!r} is not one of the classes")
        return cls(classes, background)

    def check(self, values):
        undeclared = sorted(int(v) for v in set(np.unique(values).tolist()) - set(self.classes))
        if undeclared:
            raise ValueError(f"Label values {undeclared} are not declared classes of this dataset")

    def describe(self):
        return {"unknown": 0, "background": self.background,
                "classes": {str(value): name for value, name in sorted(self.classes.items())}}


def grid_angstrom(transform):
    units = set(transform.units)
    if len(units) != 1 or next(iter(units)) not in UNIT_ANGSTROM:
        raise ValueError(f"Unsupported image units {sorted(units)}")
    factor = UNIT_ANGSTROM[next(iter(units))]
    return [float(v) * factor for v in transform.scale], [float(v) * factor for v in transform.translation]


def inferred_record(parameters, indices):
    """What inferred seeds were added: reproducible from the policy, the core version and the parameters."""
    indices = np.ascontiguousarray(np.sort(np.asarray(indices, dtype=np.int64)))
    return {"policy": SEEDING_POLICY, "implementation": SEEDING_IMPLEMENTATION,
            "core_version": package_version("cellcanvas-core"), "parameters": parameters,
            "label": parameters["background_label"], "count": int(indices.size),
            "indices_digest": "sha256:" + hashlib.sha256(indices.tobytes()).hexdigest()}


AXES = ("z", "y", "x")


def parse_region(value, shape):
    """Slices of a ``{"z": [start, stop], ...}`` request (missing axes are whole), or None for the volume."""
    if value is None:
        return None
    if not isinstance(value, dict) or not value or set(value) - set(AXES):
        raise ServiceError('region must map some of "z", "y", "x" to [start, stop].', "invalid_request", 400)
    region = []
    for axis, size in zip(AXES, shape, strict=True):
        bounds = value.get(axis, [0, size])
        if (not isinstance(bounds, list) or len(bounds) != 2
                or not all(isinstance(b, int) and not isinstance(b, bool) for b in bounds)
                or not 0 <= bounds[0] < bounds[1] <= size):
            raise ServiceError(f"region {axis} must be [start, stop] within 0..{size}.", "invalid_request", 400)
        region.append(slice(*bounds))
    return tuple(region)


def region_record(region):
    return None if region is None else {axis: [r.start, r.stop] for axis, r in zip(AXES, region, strict=True)}


def read_region(array, region):
    return array[...] if region is None else array[region]


def feature_source(values):
    """How one feature array is stored: per voxel, or as a patch-token grid interpolated on read."""
    tokens = getattr(values, "tokens", None)
    if tokens is None:
        return {"kind": "voxel", "shape": list(values.shape), "dtype": str(np.dtype(values.dtype))}
    return {"kind": "patch_grid", "shape": list(values.shape), "dtype": str(np.dtype(values.dtype)),
            "token_grid": list(tokens.shape), "interpolation": "bilinear, align_corners=false in y/x; z unchanged"}


def default_manager(dataset):
    return SemanticSegmentationManager(
        DataManager([dataset]), RandomForestClassifier(random_state=0, n_jobs=1))


class ServiceError(Exception):
    def __init__(self, message, code, status):
        super().__init__(message)
        self.code, self.status = code, status


@dataclass
class DatasetState:
    dataset: object
    manager: object
    name: str = "dataset"
    store: object = field(default_factory=MemoryStore)
    registry: object = field(default_factory=Registry)
    features: object = None
    labels: object = None
    label_policy: object = None
    model_record: dict | None = None
    trained: bool = False
    prediction_status: str = "unknown"
    painting_revision: str | None = None
    model_painting_revision: str | None = None
    prediction_version: int = 0
    prediction_painting_revision: str | None = None
    prediction_digest: str | None = None
    prediction_region: dict | None = None
    lock: object = field(default_factory=Lock)
    jobs: dict = field(default_factory=dict)
    jobs_lock: object = field(default_factory=Lock)

    def describe(self):
        dataset = self.dataset
        geometry = dataset.spatial_transform
        feature_axes = ["z", "y", "x"]
        feature_axes.insert(dataset.feature_axis % 4, "d")
        return {
            "image": {"axes": ["c", "z", "y", "x"], "shape": list(dataset.image.shape)},
            "painting": {"axes": ["z", "y", "x"], "shape": list(dataset.labels.shape),
                         "dtype": str(np.dtype(dataset.labels.dtype)), "revision": self.painting_revision},
            "prediction": {"axes": ["z", "y", "x"], "shape": list(dataset.segmentation.shape),
                           "dtype": str(np.dtype(dataset.segmentation.dtype)), "status": self.prediction_status,
                           "version": self.prediction_version,
                           "painting_revision": self.prediction_painting_revision,
                           "region": self.prediction_region},
            "features": {"axis": dataset.feature_axis, "axes": feature_axes,
                         "shapes": [list(values.shape) for values in dataset.features.values()],
                         "sources": [feature_source(values) for values in dataset.features.values()]},
            "spatial_transform": {"scale": list(geometry.scale), "translation": list(geometry.translation),
                                  "units": list(geometry.units)},
            "trained": self.trained,
            "model_painting_revision": self.model_painting_revision,
            "busy": self.lock.locked(),
            "storage": self.storage(),
            "durable": self.store.durable,
            "labels": None if self.labels is None else self.labels.describe(),
            "label_policy": self.label_policy,
            "label_policies": self.label_policies(),
            "classifiers": [c["name"] for c in self.registry.describe()["classifiers"]],
            "features_model": None if self.features is None else self.features.model_info(),
            "outputs": ["semantic"],
        }

    def label_policies(self):
        """Seeding needs core support and a declared background class."""
        if self.labels is None or self.labels.background is None or not seeding_available():
            return []
        return [SEEDING_POLICY]

    def check_grid(self, scale, translation):
        """Refuse painting made on another voxel grid (scale and origin in angstrom, z/y/x)."""
        expected = grid_angstrom(self.dataset.spatial_transform)
        for name, given, wanted in (("scale", scale, expected[0]), ("translation", translation, expected[1])):
            if given is None:
                continue
            if not isinstance(given, list) or len(given) != 3 or not np.allclose(given, wanted, rtol=1e-4, atol=1e-3):
                raise ServiceError(f"Painting {name} {given} does not match the dataset grid {wanted} (angstrom).",
                                   "geometry_mismatch", 409)

    def storage(self):
        from zarr.storage import LocalStore

        def location(array):
            if isinstance(getattr(array, "store", None), LocalStore):
                return {"uri": array.store.root.resolve().as_uri(), "path": array.path}
            return None

        return {"image": location(self.dataset.image), "painting": location(self.dataset.labels),
                "prediction": location(self.dataset.segmentation),
                "features": [location(getattr(array, "tokens", array)) for array in self.dataset.features.values()]}

    def operation(self, operation_id):
        with self.jobs_lock:
            if operation_id not in self.jobs:
                raise ServiceError("Operation not found.", "not_found", 404)
            return dict(self.jobs[operation_id])

    def submit(self, operation_id, operation, options):
        with self.jobs_lock:
            if operation_id in self.jobs:
                previous = self.jobs[operation_id]
                if (previous["operation"], previous["options"]) != (operation, options):
                    raise ServiceError("Operation ID was already used with different options.", "operation_conflict", 409)
                return dict(previous)
            if not self.lock.acquire(blocking=False):
                raise ServiceError("A dataset operation is already running.", "busy", 409)
            job = {"id": operation_id, "operation": operation, "options": options,
                   "batch_size": options.get("batch_size", 65536), "state": "running", "result": None,
                   "error": None, "accepted_at": now(), "finished_at": None}
            self.jobs[operation_id] = job
            self.persist(job)

        def run():
            try:
                result = self.execute(operation, options, reserved=True, operation_id=operation_id)
                update = {"state": "succeeded", "result": result}
            except ServiceError as error:
                update = {"state": "failed", "error": {"message": str(error), "code": error.code}}
            with self.jobs_lock:
                job.update(update, finished_at=now())
                self.persist(job)

        try:
            # Non-daemon: normal process exit waits for accepted core work.
            Thread(target=run).start()
        except Exception as error:
            self.lock.release()
            with self.jobs_lock:
                job.update(state="failed", error={"message": str(error), "code": "core_error"}, finished_at=now())
                self.persist(job)
            raise ServiceError(str(error), "core_error", 500) from error
        return self.operation(operation_id)

    def persist(self, job):
        try:
            self.store.save_operation(self.name, job)
        except Exception as error:  # a receipt that cannot be stored must not look durable
            job.update(state="failed", error={"message": f"receipt not stored: {error}", "code": "state_error"})

    def recover(self):
        """Reload durable receipts, model and prediction state after a restart; never resume interrupted work."""
        for job in self.store.operations(self.name):
            if job["state"] == "running":
                job.update(state="failed", finished_at=now(), error={
                    "message": "The service stopped before this operation finished; retry it.", "code": "interrupted"})
                self.store.save_operation(self.name, job)
            self.jobs[job["id"]] = job
        painting = self.store.record("painting", self.name)
        if painting and painting["digest"] == label_digest(self.dataset.labels[...]):
            self.painting_revision = painting["revision"]
        model, record = self.store.load_model(self.name)
        if model is not None and hasattr(self.manager, "model"):
            self.manager.model = model
            self.trained, self.model_record = True, record
            self.model_painting_revision = record["painting_revision"]
        prediction = self.store.record("prediction", self.name)
        if prediction:
            # An unfinished or failed write never published its version.
            completed = prediction["status"] == "ready"
            self.prediction_version = prediction["version"] - (0 if completed else 1)
            self.prediction_painting_revision = prediction.get("painting_revision")
            region = parse_region(prediction.get("region"), self.dataset.segmentation.shape)
            if prediction["status"] == "writing":
                self.prediction_status = "failed"
            elif prediction["status"] == "ready" and prediction["digest"] == label_digest(
                    read_region(self.dataset.segmentation, region)):
                self.prediction_status, self.prediction_digest = "ready", prediction["digest"]
                self.prediction_region = prediction.get("region")
            else:
                self.prediction_status = "unknown"

    def execute(self, operation, options=None, *, reserved=False, operation_id=None):
        # Acquire and release inside the worker, so request cancellation cannot
        # release the model/storage lock while core is still running.
        options = options or {}
        if not reserved and not self.lock.acquire(blocking=False):
            raise ServiceError("A dataset operation is already running.", "busy", 409)
        try:
            if operation == "train":
                self.train(options, operation_id)
            else:
                self.predict(options.get("batch_size", 65536), operation_id, options.get("region"))
            return {"status": "success", "operation": operation, "trained": self.trained,
                    "prediction_status": self.prediction_status, "prediction_version": self.prediction_version,
                    "painting_revision": self.model_painting_revision,
                    "inferred": (self.model_record or {}).get("inferred", [])}
        except ServiceError:
            raise
        except Exception as error:
            # Expected core failures are handled responses, so clients can
            # inspect state or retry on the same HTTP connection.
            raise ServiceError(str(error), "core_error", 500) from error
        finally:
            self.lock.release()

    def train(self, options, operation_id):
        policy = self.resolve_policy(options.get("label_policy", DEFAULT))
        classifier = options.get("classifier")
        self.trained = False
        self.model_painting_revision = None
        self.prediction_status = "unknown"
        revision = self.painting_revision
        manager = self.manager
        if policy is not None or classifier is not None:
            from cellcanvas_core.data.data_manager import DataManager
            from sklearn.base import clone

            if classifier is not None:
                try:
                    model = self.registry.classifier(classifier)
                except KeyError:
                    raise ServiceError(f"Classifier {classifier!r} is not available.", "invalid_request", 400) from None
            else:
                model = clone(self.manager.model)
            if policy is not None:
                from cellcanvas_core.data.seeding import SeededDataManager

                data = SeededDataManager([self.dataset], **policy[1])
            else:
                data = DataManager([self.dataset])
            manager = SemanticSegmentationManager(data, model)
        before = label_digest(self.dataset.labels[...])
        manager.fit()
        if label_digest(self.dataset.labels[...]) != before:
            raise ServiceError("Painting changed while training; seeds must never be written.", "painting_changed", 500)
        inferred = []
        if policy is not None:
            inferred = [inferred_record(policy[1], manager.data.seed_indices(self.dataset))]
        self.manager = manager
        self.trained = True
        self.model_painting_revision = revision
        record = {"operation_id": operation_id, "painting_revision": revision, "painting_digest": before,
                  "classifier": classifier or type(getattr(manager, "model", manager)).__name__,
                  "label_policy": None if policy is None else {"policy": policy[0], "parameters": policy[1]},
                  "inferred": inferred, "trained_at": now(),
                  "features": None if self.features is None else self.features.model_info()}
        self.model_record = record
        if hasattr(manager, "model"):
            self.store.save_model(self.name, manager.model, record)

    def predict(self, batch_size, operation_id, region=None):
        if not self.trained:
            raise ServiceError("Train this dataset before predicting.", "not_trained", 409)
        slices = parse_region(region, self.dataset.segmentation.shape)
        version = self.prediction_version + 1
        self.prediction_status = "writing"
        self.prediction_region = None
        self.store.save_record("prediction", self.name, {"status": "writing", "version": version,
                                                         "operation_id": operation_id, "region": region})
        try:
            # Whole-volume calls keep the original signature, so cores without regions still serve them.
            self.manager.predict_to_segmentation(self.dataset, batch_size=batch_size,
                                                 **({} if slices is None else {"region": slices}))
            # Digest what is stored, not what was computed, so the record describes the readable output.
            digest = label_digest(read_region(self.dataset.segmentation, slices))
            self.store.save_result(self.name, version, self.result_record(version, operation_id, digest, region))
        except Exception:
            self.prediction_status = "failed"
            self.store.save_record("prediction", self.name, {"status": "failed", "version": version,
                                                             "operation_id": operation_id})
            raise
        self.prediction_version = version
        self.prediction_painting_revision = self.model_painting_revision
        self.prediction_status = "ready"
        self.prediction_digest = digest
        self.prediction_region = region
        self.store.save_record("prediction", self.name, {
            "status": "ready", "version": version, "operation_id": operation_id, "digest": digest,
            "painting_revision": self.prediction_painting_revision, "region": region})

    def result_record(self, version, operation_id, digest, region=None):
        """What a completed prediction was made from; a client checks ``digest`` against the voxels it read.

        ``shape`` is the dataset grid; a region prediction covers ``region`` only and its digest is of those voxels.
        """
        model = self.model_record or {}
        scale, translation = grid_angstrom(self.dataset.spatial_transform)
        return {"version": version, "operation_id": operation_id, "digest": digest, "encoding": "semantic",
                "shape": list(self.dataset.segmentation.shape), "region": region, "scale_angstrom": scale,
                "translation_angstrom": translation, "backend": backend_version(),
                "painting_revision": model.get("painting_revision"), "painting_digest": model.get("painting_digest"),
                "classifier": model.get("classifier"), "label_policy": model.get("label_policy"),
                "inferred": model.get("inferred", []),
                "features": None if self.features is None else self.features.provenance(),
                "labels": None if self.labels is None else self.labels.describe(), "created_at": now()}

    def resolve_policy(self, request):
        """(policy, parameters) of a train request, or None for painted labels only.

        ``{"policy": "inferred_background", "parameters": {...}}`` adds inferred background seeds; ``null``
        disables them. background_label is the dataset's declared background class and cannot be overridden.
        """
        if request is DEFAULT:
            request = self.label_policy
        if request is None:
            return None
        if not isinstance(request, dict) or set(request) - {"policy", "parameters"}:
            raise ServiceError("label_policy must be null or {policy, parameters}", "invalid_request", 400)
        policy = request.get("policy")
        if policy is None:
            return None
        if policy not in self.label_policies():
            raise ServiceError(f"Label policy {policy!r} is not available for this dataset", "invalid_request", 400)
        parameters = dict(request.get("parameters") or {})
        for key, value in parameters.items():
            kind = SEEDING_PARAMETERS.get(key)
            if kind is None or isinstance(value, bool) or not isinstance(value, (int, float)) or (
                    kind is int and not isinstance(value, int)):
                raise ServiceError(f"Unsupported {policy} parameter {key}={value!r}", "invalid_request", 400)
        parameters["background_label"] = self.labels.background
        return policy, parameters

    def reserve(self):
        if not self.lock.acquire(blocking=False):
            raise ServiceError("A dataset operation is already running.", "busy", 409)

    def replace_painting(self, revision, shape, indices, values, scale=None, translation=None):
        """Replace all painting with sparse labels; zero (unknown) elsewhere."""
        labels = self.dataset.labels
        if list(shape) != list(labels.shape):
            raise ServiceError(f"Painting shape {list(shape)} does not match {list(labels.shape)}.",
                               "geometry_mismatch", 409)
        self.check_grid(scale, translation)
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=np.int64)
        size = int(np.prod(labels.shape))
        if indices.shape != values.shape or indices.ndim != 1:
            raise ServiceError("indices and values must be equal-length lists.", "invalid_request", 400)
        if indices.size and (indices.min() < 0 or indices.max() >= size or len(np.unique(indices)) != indices.size):
            raise ServiceError("indices must be unique flat C-order voxel indices.", "invalid_request", 400)
        dtype = np.dtype(labels.dtype)
        if values.size and (values.min() < 1 or values.max() > np.iinfo(dtype).max):
            raise ServiceError("Painted values must be positive class IDs; omit unannotated voxels.",
                               "invalid_request", 400)
        if self.labels is not None and values.size:
            try:
                self.labels.check(values)
            except ValueError as error:
                raise ServiceError(str(error), "invalid_request", 400) from None
        self.reserve()
        try:
            labels[...] = 0
            if indices.size:
                labels.set_coordinate_selection(np.unravel_index(indices, labels.shape), values.astype(dtype))
            self.painting_revision = revision
            self.store.save_record("painting", self.name, {"revision": revision,
                                                           "digest": label_digest(labels[...])})
            return {"painting_revision": revision, "voxels": int(indices.size)}
        except Exception as error:
            self.painting_revision = None
            raise ServiceError(str(error), "core_error", 500) from error
        finally:
            self.lock.release()

    def read_prediction(self, version):
        self.reserve()
        try:
            if self.prediction_status != "ready" or version != self.prediction_version:
                raise ServiceError("That prediction version is not complete and current.", "not_ready", 409)
            array = self.dataset.segmentation
            region = parse_region(self.prediction_region, array.shape)
            data = np.ascontiguousarray(read_region(array, region))
            if self.prediction_digest is not None and label_digest(data) != self.prediction_digest:
                self.prediction_status = "unknown"
                raise ServiceError("Stored prediction changed after it was completed.", "not_ready", 409)
            bounds = region or tuple(slice(0, n) for n in array.shape)
            return data, {"X-Shape": ",".join(map(str, data.shape)), "X-Dtype": str(data.dtype),
                          "X-Region": ",".join(f"{r.start}:{r.stop}" for r in bounds),
                          "X-Prediction-Version": str(version),
                          "X-Painting-Revision": self.prediction_painting_revision or "",
                          "X-Result-Digest": self.prediction_digest or label_digest(data)}
        finally:
            self.lock.release()


async def options(request, allowed):
    try:
        body = await request.body()
        data = await request.json() if body else {}
    except ValueError:
        raise ServiceError("Request body must be a JSON object.", "invalid_request", 400) from None
    if not isinstance(data, dict) or set(data) - allowed:
        raise ServiceError("Request contains unsupported options.", "invalid_request", 400)
    return data


def create_app(datasets, *, manager_factory=default_manager, store=None, registry=None, dataset_options=None):
    """Register server-owned DataSet objects; model construction is injectable.

    The service owns each manager and serializes its operations. Callers must
    serialize external access to the same model, painting or prediction stores.

    ``store`` (state.DirectoryStore) makes receipts, models and results durable.
    ``dataset_options`` maps a dataset name to ``features_manifest`` (saved
    feature provenance), ``labels`` (declared classes; see Labels) and
    ``label_policy`` (default training label policy, e.g. inferred
    background seeds).
    """
    if not datasets or any(not isinstance(name, str) or not name or "/" in name for name in datasets):
        raise ValueError("Register at least one dataset with nonempty names containing no slash.")
    store = store or MemoryStore()
    registry = registry or Registry()
    dataset_options = dataset_options or {}
    states = {}
    for name, dataset in datasets.items():
        extra = dict(dataset_options.get(name) or {})
        unknown = set(extra) - {"features_manifest", "features_provider", "labels", "label_policy"}
        if unknown:
            raise ValueError(f"Dataset {name}: unsupported options {sorted(unknown)}")
        features = None
        if extra.get("features_manifest") is not None:
            features = FeatureSource(extra["features_manifest"], extra.get("features_provider", "saved"))
            registry.check_features(features)
        labels = None if extra.get("labels") is None else Labels.from_config(extra["labels"])
        state = DatasetState(dataset, manager_factory(dataset), name=name, store=store, registry=registry,
                             features=features, labels=labels, label_policy=extra.get("label_policy"))
        if state.label_policy is not None:
            try:
                state.resolve_policy(state.label_policy)
            except ServiceError as error:
                raise ValueError(f"Dataset {name}: {error}") from None
        state.recover()
        states[name] = state
    service_id = store.service_id

    def metadata(state):
        return dict(state.describe(), service_id=service_id)

    def state_for(request):
        try:
            return states[request.path_params["name"]]
        except KeyError:
            raise ServiceError("Dataset not found.", "not_found", 404) from None

    async def health(request):
        return JSONResponse({"status": "ok"})

    async def list_datasets(request):
        return JSONResponse({"datasets": {name: metadata(state) for name, state in states.items()}})

    async def describe(request):
        return JSONResponse(metadata(state_for(request)))

    def operation_id(data):
        if "operation_id" not in data and "service_id" not in data:
            return None
        if data.get("service_id") != service_id:
            raise ServiceError("Service instance changed; reconnect before starting work.", "service_changed", 409)
        try:
            return UUID(data["operation_id"]).hex
        except (KeyError, TypeError, ValueError, AttributeError):
            raise ServiceError("operation_id must be a UUID string.", "invalid_request", 400) from None

    async def operation_status(request):
        state = state_for(request)
        return JSONResponse(dict(state.operation(request.path_params["operation_id"]), service_id=service_id))

    async def train(request):
        state = state_for(request)
        data = await options(request, {"operation_id", "service_id", "classifier", "label_policy"})
        identifier = operation_id(data)
        settings = {key: data[key] for key in ("classifier", "label_policy") if key in data}
        if "classifier" in settings and not isinstance(settings["classifier"], str):
            raise ServiceError("classifier must be a registered classifier name.", "invalid_request", 400)
        if "label_policy" in settings:
            state.resolve_policy(settings["label_policy"])
        if identifier is not None:
            return JSONResponse(dict(state.submit(identifier, "train", settings), service_id=service_id),
                                status_code=202)
        result = await run_in_threadpool(state.execute, "train", settings)
        return JSONResponse(result)

    async def predict(request):
        state = state_for(request)
        data = await options(request, {"batch_size", "operation_id", "service_id", "region"})
        batch_size = data.get("batch_size", 65536)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ServiceError("batch_size must be a positive integer.", "invalid_request", 400)
        settings = {"batch_size": batch_size}
        if data.get("region") is not None:
            parse_region(data["region"], state.dataset.segmentation.shape)
            settings["region"] = data["region"]
        identifier = operation_id(data)
        if identifier is not None:
            return JSONResponse(dict(state.submit(identifier, "predict", settings),
                                     service_id=service_id), status_code=202)
        result = await run_in_threadpool(state.execute, "predict", settings)
        return JSONResponse(result)

    def check_service(data):
        if data.get("service_id") != service_id:
            raise ServiceError("Service instance changed; reconnect before starting work.", "service_changed", 409)

    async def painting(request):
        state = state_for(request)
        data = await options(request, {"service_id", "revision", "shape", "indices", "values", "scale",
                                       "translation"})
        check_service(data)
        revision = data.get("revision")
        if not isinstance(revision, str) or not revision or not isinstance(data.get("shape"), list):
            raise ServiceError("revision (string) and shape are required.", "invalid_request", 400)
        result = await run_in_threadpool(state.replace_painting, revision, data["shape"],
                                         data.get("indices", []), data.get("values", []), data.get("scale"),
                                         data.get("translation"))
        return JSONResponse(dict(result, service_id=service_id))

    async def prediction(request):
        state = state_for(request)
        try:
            version = int(request.query_params["version"])
        except (KeyError, ValueError):
            raise ServiceError("version query parameter is required.", "invalid_request", 400) from None
        data, headers = await run_in_threadpool(state.read_prediction, version)
        headers["X-Service-Id"] = service_id
        return Response(data.tobytes(), media_type="application/octet-stream", headers=headers)

    async def providers(request):
        return JSONResponse(dict(registry.describe(), service_id=service_id, durable=store.durable))

    async def model(request):
        state = state_for(request)
        if state.model_record is None:
            raise ServiceError("This dataset has no trained model.", "not_trained", 404)
        return JSONResponse(dict(state.model_record, trained=state.trained, service_id=service_id))

    def inferred_seeds(state):
        state.reserve()
        try:
            record = state.model_record or {}
            if not record.get("inferred"):
                raise ServiceError("The current model was trained without inferred labels.", "not_found", 404)
            from cellcanvas_core.data.seeding import SeededDataManager

            policy = record["label_policy"]
            indices = SeededDataManager([state.dataset], **policy["parameters"]).seed_indices(state.dataset)
            document = inferred_record(policy["parameters"], indices)
            if document["indices_digest"] != record["inferred"][0]["indices_digest"]:
                raise ServiceError("Seeds no longer reproduce; features or painting changed since training.",
                                   "not_reproducible", 409)
            return {"inferred": document, "shape": list(state.dataset.labels.shape),
                    "indices": [int(i) for i in indices]}
        finally:
            state.lock.release()

    async def inferred(request):
        state = state_for(request)
        return JSONResponse(dict(await run_in_threadpool(inferred_seeds, state), service_id=service_id))

    async def result(request):
        state = state_for(request)
        try:
            version = int(request.path_params["version"])
        except ValueError:
            raise ServiceError("version must be an integer.", "invalid_request", 400) from None
        record = state.store.result(state.name, version)
        if record is None:
            raise ServiceError("No result record for that version.", "not_found", 404)
        current = state.prediction_status == "ready" and version == state.prediction_version
        return JSONResponse({"result": record, "current": current, "service_id": service_id})

    async def service_error(request, error):
        return JSONResponse({"error": str(error), "code": error.code}, status_code=error.status)

    app = Starlette(routes=[
        Route("/health", health),
        Route("/datasets", list_datasets),
        Route("/datasets/{name}", describe),
        Route("/datasets/{name}/train", train, methods=["POST"]),
        Route("/datasets/{name}/predict", predict, methods=["POST"]),
        Route("/datasets/{name}/operations/{operation_id}", operation_status),
        Route("/datasets/{name}/painting", painting, methods=["PUT"]),
        Route("/datasets/{name}/prediction", prediction),
        Route("/datasets/{name}/model", model),
        Route("/datasets/{name}/inferred", inferred),
        Route("/datasets/{name}/results/{version}", result),
        Route("/providers", providers),
    ], exception_handlers={ServiceError: service_error})
    app.state.datasets = states
    return app
