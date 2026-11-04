"""Launch the core service using server-side dataset paths."""

import argparse
import json
from pathlib import Path

import uvicorn
from cellcanvas_core.data.data_set import DataSet

from .app import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Config: {"datasets": {"demo": {"image_path": "image.zarr", '
        '"features_path": "features.zarr", "labels_path": "painting.zarr", '
        '"segmentation_path": "prediction.zarr"}}}. Paths use the working directory.'))
    parser.add_argument("config", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        if not isinstance(config, dict) or set(config) != {"datasets"} or not isinstance(config["datasets"], dict):
            raise ValueError("Config must contain a datasets object.")
        datasets = {name: DataSet.from_paths(**paths) for name, paths in config["datasets"].items()}
        app = create_app(datasets)
    except Exception as error:
        parser.error(str(error))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
