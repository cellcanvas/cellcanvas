"""Registering a copick-style OME-Zarr 2 tomogram and saved features as a core dataset."""

import json

import pytest

pytest.importorskip("cellcanvas_core", reason="Requires the service extra")

import numpy as np
import zarr
from cellcanvas_core.data.data_set import DataSet

from cellcanvas_service.prepare import prepare


@pytest.fixture
def source(tmp_path):
    shape = (4, 5, 6)
    tomogram = tmp_path / "tomo.zarr"
    group = zarr.open_group(str(tomogram), mode="w", zarr_format=2)
    group.create_array("0", data=np.arange(120, dtype="f4").reshape(shape))
    group.attrs["multiscales"] = [{
        "axes": [{"name": axis, "type": "space", "unit": "angstrom"} for axis in "zyx"],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [10.0, 10.0, 10.0]}, {"type": "translation", "translation": [0, 0, 0]}]}],
    }]
    features = tmp_path / "features.zarr"
    zarr.save_array(str(features), np.ones((3, *shape), "f4"), zarr_format=2)
    manifest = {"model": "fixture", "license": "CC0-1.0", "public": True, "source": str(tomogram),
                "grid": {"shape": list(shape)}}
    return tmp_path, tomogram, features, manifest


def write(tmp_path, manifest):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_prepare_preserves_geometry_and_opens_in_core(source):
    tmp_path, tomogram, features, manifest = source
    config = prepare(tomogram, features, write(tmp_path, manifest), tmp_path / "out", "ts1")
    paths = json.loads(config.read_text())["datasets"]["ts1"]
    dataset = DataSet.from_paths(**paths)
    assert dataset.labels.shape == (4, 5, 6)
    assert list(dataset.spatial_transform.scale) == [10.0, 10.0, 10.0]
    assert list(dataset.spatial_transform.translation) == [0.0, 0.0, 0.0]
    assert set(dataset.spatial_transform.units) == {"angstrom"}
    np.testing.assert_array_equal(dataset.image[0], zarr.open_group(str(tomogram), mode="r")["0"][:])


@pytest.mark.parametrize("change", [{"public": False}, {"grid": {"shape": [4, 5, 7]}}, {"license": None}])
def test_prepare_refuses_private_or_misaligned_features(source, change):
    tmp_path, tomogram, features, manifest = source
    manifest = {k: v for k, v in dict(manifest, **change).items() if v is not None}
    with pytest.raises(ValueError):
        prepare(tomogram, features, write(tmp_path, manifest), tmp_path / "out", "ts1")
