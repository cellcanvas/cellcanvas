"""Launch the core service using server-side dataset paths."""

import argparse
import json
from pathlib import Path

import uvicorn
from cellcanvas_core.data.data_set import DataSet

from .app import create_app, manager_factory_for


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Config: {"datasets": {"demo": {"image_path": "image.zarr", '
        '"features_path": "features.zarr", "labels_path": "painting.zarr", '
        '"segmentation_path": "prediction.zarr"}}, "model": "rf"}. Paths use the working '
        'directory; "model" is optional: "rf" (random forest, the default) or "fewshot" '
        '(core\'s FewShotClassifier).'))
    parser.add_argument("config", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    try:
        app = app_from_config(json.loads(args.config.read_text()))
    except Exception as error:
        parser.error(str(error))
    uvicorn.run(app, host=args.host, port=args.port)


def app_from_config(config):
    """The service app of a parsed config: its datasets and, optionally, its model."""
    if not isinstance(config, dict) or not isinstance(config.get("datasets"), dict) \
            or set(config) - {"datasets", "model"}:
        raise ValueError('Config must contain a "datasets" object and at most a "model" name.')
    factory = manager_factory_for(config.get("model", "rf"))
    datasets = {name: DataSet.from_paths(**paths) for name, paths in config["datasets"].items()}
    return create_app(datasets, manager_factory=factory)


if __name__ == "__main__":
    main()
