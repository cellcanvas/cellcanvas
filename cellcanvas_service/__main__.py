"""Launch the core service using server-side dataset paths.

    python -m cellcanvas_service service.json [--state-dir DIR] [--profile profiles/local.json]

A profile (see ``cellcanvas_service.profiles``) supplies the bind address, data root and storage options, state
directory, provider plugins and prompt backends for a deployment; command-line options override it.
"""

import argparse
import json
import os
from pathlib import Path

DATASET_OPTIONS = ("features_manifest", "features_provider", "labels", "label_policy")


def load(config_path, profile=None, state_dir=None):
    """Build the Starlette app from a service config and an optional deployment profile."""
    from cellcanvas_core.data.data_set import DataSet

    from .app import create_app
    from .profiles import dataset_arguments
    from .providers import Registry
    from .state import open_store

    config = json.loads(Path(config_path).read_text())
    if not isinstance(config, dict) or set(config) != {"datasets"} or not isinstance(config["datasets"], dict):
        raise ValueError("Config must contain a datasets object.")
    providers = (profile or {}).get("providers", {})
    registry = Registry(allow_private=bool(providers.get("allow_private")),
                        prompt_backends=list((profile or {}).get("prompt_backends", [])))
    registry.load_plugins(providers.get("plugins", []))
    datasets, options = {}, {}
    for name, entry in config["datasets"].items():
        entry = dict(entry)
        options[name] = {key: entry.pop(key) for key in DATASET_OPTIONS if key in entry}
        datasets[name] = DataSet.from_paths(**dataset_arguments(entry, profile))
    state_dir = state_dir or (profile or {}).get("state_dir")
    return create_app(datasets, store=open_store(state_dir), registry=registry, dataset_options=options)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=(
        'Config: {"datasets": {"demo": {"image_path": "image.zarr", '
        '"features_path": "features.zarr", "labels_path": "painting.zarr", '
        '"segmentation_path": "prediction.zarr", "features_manifest": {...}, '
        '"labels": {"classes": {"1": "background", "2": "membrane"}, "background": 1}, '
        '"label_policy": null}}}. Relative paths use the '
        'profile data root, else the working directory.'))
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    try:
        from .profiles import load_profile

        profile = load_profile(args.profile) if args.profile else None
        config = args.config or (profile or {}).get("config")
        if not config:
            raise ValueError("Give a service config or a profile with a config.")
        threads = (profile or {}).get("compute", {}).get("threads")
        if threads:
            for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(variable, str(threads))
        app = load(config, profile, args.state_dir)
    except Exception as error:
        parser.error(str(error))
    import uvicorn

    bind = (profile or {}).get("bind", {})
    uvicorn.run(app, host=args.host or bind.get("host", "127.0.0.1"), port=args.port or bind.get("port", 8001))


if __name__ == "__main__":
    main()
