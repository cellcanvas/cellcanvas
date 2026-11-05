"""Register one copick reconstruction and its saved features as a core dataset.

The copick tomogram (OME-Zarr 2, z/y/x in Å) is copied into a Zarr 3 OME image with a channel axis and the same
scale, origin and units, so the core does not depend on copick or Zarr 2. Features must already exist on the
reconstruction's voxel grid (``d, z, y, x`` or ``z, y, x, d``); a model with a patch stride needs its features
resampled to that grid by its producer, with the resampling recorded in the manifest. The features manifest must
name a public model (``"public": true``); private models belong to private adapters and are refused here.

    python -m cellcanvas_service.prepare TOMOGRAM.zarr FEATURES.zarr MANIFEST.json OUT_DIR --name ts1

writes ``OUT_DIR/service.json`` for ``python -m cellcanvas_service`` and ``OUT_DIR/geometry.json``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

REQUIRED_MANIFEST = ("model", "license", "public", "source", "grid")
UNIT_ANGSTROM = {"angstrom": 1.0, "nanometer": 10.0}


def tomogram_geometry(path):
    group = zarr.open_group(str(path), mode="r")
    attrs = group.attrs.asdict()
    multiscales = attrs.get("multiscales") or attrs.get("ome", {}).get("multiscales")
    if not multiscales:
        raise ValueError("Tomogram is not an OME-Zarr multiscale image")
    ms = multiscales[0]
    axes = ms["axes"]
    names = [a["name"] for a in axes]
    if names[-3:] != ["z", "y", "x"] or len(names) != 3:
        raise ValueError(f"Expected z, y, x tomogram axes, found {names}")
    units = {a.get("unit", "angstrom") for a in axes}
    if len(units) != 1 or next(iter(units)) not in UNIT_ANGSTROM:
        raise ValueError(f"Unsupported units {sorted(units)}")
    level = ms["datasets"][0]
    scale, translation = [1.0] * 3, [0.0] * 3
    for transform in level.get("coordinateTransformations", []):
        if transform["type"] == "scale":
            scale = [float(v) for v in transform["scale"]]
        elif transform["type"] == "translation":
            translation = [float(v) for v in transform["translation"]]
    return group[level["path"]], {"scale": scale, "translation": translation, "unit": next(iter(units))}


def validate_manifest(manifest, shape):
    missing = [key for key in REQUIRED_MANIFEST if key not in manifest]
    if missing:
        raise ValueError(f"Features manifest lacks {missing}")
    if manifest["public"] is not True:
        raise ValueError("Features manifest does not declare a public model; use a private adapter instead")
    grid = manifest["grid"]
    if list(grid.get("shape", [])) != list(shape):
        raise ValueError(f"Features grid {grid.get('shape')} differs from reconstruction {list(shape)}")


def prepare(tomogram, features, manifest_path, out, name):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source, geometry = tomogram_geometry(tomogram)
    shape = list(source.shape)
    manifest = json.loads(Path(manifest_path).read_text())
    validate_manifest(manifest, shape)
    feature_array = zarr.open_array(str(features), mode="r")
    if list(feature_array.shape[1:]) == shape:
        axis = 0
    elif list(feature_array.shape[:3]) == shape:
        axis = -1
    else:
        raise ValueError(f"Feature shape {list(feature_array.shape)} does not contain the grid {shape}")
    image_path = out / "image.zarr"
    group = zarr.open_group(str(image_path), mode="w", zarr_format=3)
    group.create_array("0", data=np.asarray(source[...])[None], dimension_names=("c", "z", "y", "x"))
    group.attrs["ome"] = {"version": "0.5", "multiscales": [{
        "axes": [{"name": "c", "type": "channel"}]
        + [{"name": axis_name, "type": "space", "unit": geometry["unit"]} for axis_name in ("z", "y", "x")],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, *geometry["scale"]]},
            {"type": "translation", "translation": [0.0, *geometry["translation"]]},
        ]}],
    }]}
    paths = {"image_path": str(image_path), "features_path": str(Path(features).resolve()),
             "labels_path": str(out / "painting.zarr"), "segmentation_path": str(out / "prediction.zarr"),
             "make_missing_datasets": True, "feature_axis": axis}
    (out / "service.json").write_text(json.dumps({"datasets": {name: paths}}, indent=2))
    (out / "geometry.json").write_text(json.dumps({"shape": shape, **geometry, "source": str(tomogram),
                                                   "features_manifest": manifest}, indent=2))
    return out / "service.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tomogram")
    parser.add_argument("features")
    parser.add_argument("manifest")
    parser.add_argument("out")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    try:
        print(prepare(args.tomogram, args.features, args.manifest, args.out, args.name))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
