"""A dataset whose features are a stored patch-token grid, served with region predictions."""

import json
import socket
import time
from threading import Thread

import numpy as np
import pytest

patch_grid = pytest.importorskip("cellcanvas_core.data.patch_grid",
                                 reason="Token grids need cellcanvas-core with patch-grid features")

import httpx  # noqa: E402
import uvicorn  # noqa: E402
import zarr  # noqa: E402
from cellcanvas_core.data.data_manager import DataManager  # noqa: E402
from cellcanvas_core.data.data_set import DataSet  # noqa: E402
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from cellcanvas_service.__main__ import load  # noqa: E402

SHAPE = (3, 9, 11)
MANIFEST = {"model": "test tokens", "license": "CC0-1.0", "public": True, "source": "synthetic",
            "grid": {"resampling": "4x5 patch tokens, bilinear on read, align_corners=false; z unchanged"}}


@pytest.fixture
def config(tmp_path):
    image = zarr.open_group(str(tmp_path / "image"), mode="w", zarr_format=3)
    image.create_array("0", data=np.zeros((1, *SHAPE), "f4"), dimension_names=("c", "z", "y", "x"))
    image.attrs["ome"] = {"version": "0.5", "multiscales": [{
        "axes": [{"name": "c", "type": "channel"}] + [
            {"name": axis, "type": "space", "unit": "angstrom"} for axis in ("z", "y", "x")],
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 10, 10, 10]}]}],
    }]}
    tokens = np.random.default_rng(0).normal(size=(6, SHAPE[0], 4, 5)).astype("f4")
    patch_grid.write_patch_grid(str(tmp_path / "tokens.zarr"), tokens, SHAPE, "float16", {"model": "test tokens"})
    entry = {"image_path": str(tmp_path / "image"), "features_path": str(tmp_path / "tokens.zarr"),
             "labels_path": str(tmp_path / "painting"), "segmentation_path": str(tmp_path / "prediction"),
             "make_missing_datasets": True, "features_manifest": MANIFEST,
             "labels": {"classes": {"1": "background", "2": "membrane"}, "background": 1}}
    path = tmp_path / "service.json"
    path.write_text(json.dumps({"datasets": {"demo": entry}}))
    return path


def serve(app, servers):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    servers.append((server, thread, sock))
    deadline = time.monotonic() + 5
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline
        time.sleep(.01)
    return f"http://127.0.0.1:{sock.getsockname()[1]}"


@pytest.fixture
def servers():
    running = []
    yield running
    for server, thread, sock in reversed(running):
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


def test_token_grid_dataset_trains_and_predicts_one_slice(config, tmp_path, servers):
    url = serve(load(config, state_dir=tmp_path / "state"), servers)
    painting = np.zeros(SHAPE, "i4")
    painting[1, 1:4, 1:4], painting[1, 6:8, 6:10], painting[2, 4, 2:8] = 1, 2, 2
    indices = np.flatnonzero(painting)
    with httpx.Client(base_url=url, timeout=30) as client:
        meta = client.get("/datasets/demo").json()
        (source,) = meta["features"]["sources"]
        assert source["kind"] == "patch_grid" and source["token_grid"] == [6, 3, 4, 5]
        assert source["shape"] == [6, *SHAPE] and source["dtype"] == "float16"
        assert meta["storage"]["features"][0]["uri"].endswith("tokens.zarr")
        body = {"service_id": meta["service_id"], "revision": "r1", "shape": list(SHAPE),
                "indices": indices.tolist(), "values": painting.reshape(-1)[indices].tolist()}
        assert client.put("/datasets/demo/painting", json=body).status_code == 200
        assert client.post("/datasets/demo/train", json={}).json()["status"] == "success"
        for bad in ({"z": [0, 9]}, {"z": [2, 2]}, {"t": [0, 1]}, {"y": [0]}, []):
            response = client.post("/datasets/demo/predict", json={"region": bad})
            assert response.status_code == 400, bad
        result = client.post("/datasets/demo/predict", json={"region": {"z": [1, 2]}}).json()
        assert result["prediction_version"] == 1
        assert client.get("/datasets/demo").json()["prediction"]["region"] == {"z": [1, 2]}
        response = client.get("/datasets/demo/prediction", params={"version": 1})
        assert response.headers["x-region"] == "1:2,0:9,0:11"
        assert response.headers["x-shape"] == "1,9,11"
        region = np.frombuffer(response.content, response.headers["x-dtype"]).reshape(1, 9, 11)
        record = client.get("/datasets/demo/results/1").json()["result"]
        assert record["region"] == {"z": [1, 2]} and record["shape"] == list(SHAPE)
        assert record["features"]["model"] == "test tokens"

    # Same painting and default classifier directly in core on the dense expansion of the tokens.
    stored = zarr.open_array(str(tmp_path / "tokens.zarr"), mode="r")
    dense = patch_grid.PatchGridFeatures.from_zarr(stored)[...]
    reference = DataSet(np.zeros((1, *SHAPE)), {"dense": dense}, painting, np.zeros(SHAPE, "i4"))
    manager = SemanticSegmentationManager(DataManager([reference]), RandomForestClassifier(random_state=0, n_jobs=1))
    manager.fit()
    manager.predict_to_segmentation(reference)
    np.testing.assert_array_equal(region[0], reference.segmentation[1])
    assert set(np.unique(region)) == {1, 2}
    assert not zarr.open_array(str(tmp_path / "prediction"), mode="r")[[0, 2]].any()

    # A restarted service still serves that completed region version.
    servers[-1][0].should_exit = True
    url = serve(load(config, state_dir=tmp_path / "state"), servers)
    with httpx.Client(base_url=url, timeout=30) as client:
        meta = client.get("/datasets/demo").json()
        assert meta["prediction"]["status"] == "ready" and meta["prediction"]["region"] == {"z": [1, 2]}
        response = client.get("/datasets/demo/prediction", params={"version": 1})
        assert response.status_code == 200 and response.content == region.tobytes()
        # A whole-volume prediction is a new version over every voxel.
        assert client.post("/datasets/demo/predict", json={}).json()["prediction_version"] == 2
        response = client.get("/datasets/demo/prediction", params={"version": 2})
        assert response.headers["x-region"] == "0:3,0:9,0:11"
        whole = np.frombuffer(response.content, response.headers["x-dtype"]).reshape(SHAPE)
        np.testing.assert_array_equal(whole, reference.segmentation[:])
