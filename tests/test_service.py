"""Compare real HTTP operations with direct core execution on Zarr 3 stores."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest

pytest.importorskip("cellcanvas_core", reason="Core service tests require the service extra")

import httpx
import numpy as np
import uvicorn
import zarr
from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.data.data_set import DataSet
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager
from sklearn.tree import DecisionTreeClassifier

from cellcanvas_service.app import create_app, default_manager


@pytest.fixture
def case_factory(tmp_path):
    cases = []

    def make(depth=1, feature_axis=0, painted=True):
        root = tmp_path / str(len(cases))
        root.mkdir()
        indices = np.arange(depth * 5 * 4).reshape(depth, 5, 4)
        paths = {key: str(root / name) for key, name in (
            ("image_path", "image"), ("features_path", "features"),
            ("labels_path", "painting"), ("segmentation_path", "prediction"))}
        image = np.stack([indices, indices + 100]).astype("f4")
        group = zarr.open_group(paths["image_path"], mode="w", zarr_format=3)
        group.create_array("0", data=image, dimension_names=("c", "z", "y", "x"))
        group.attrs["ome"] = {"version": "0.5", "multiscales": [{
            "axes": [{"name": "c", "type": "channel"}] + [
                {"name": axis, "type": "space", "unit": "micrometer"} for axis in ("z", "y", "x")],
            "datasets": [{"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [1, 3, 2, .5]},
                {"type": "translation", "translation": [0, 7, 11, 13]},
            ]}],
        }]}
        features = (indices % 2)[None].astype("f4")
        if feature_axis == -1:
            features = np.moveaxis(features, 0, -1)
        zarr.save_array(paths["features_path"], features, zarr_format=3)
        options = dict(paths, make_missing_datasets=True, feature_axis=feature_axis)
        dataset = DataSet.from_paths(**options)
        if painted:
            dataset.labels[0, 2, 0] = 1
            dataset.labels[0, 2, 1] = 3
        result = SimpleNamespace(dataset=dataset, options=options, root=root, image=image,
                                 features=features, expected=indices % 2 * 2 + 1)
        cases.append(result)
        return result
    return make


@pytest.fixture
def serve():
    servers = []

    def start(datasets, **kwargs):
        app = create_app(datasets, **kwargs)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        thread = Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        servers.append((server, thread, sock))
        deadline = time.monotonic() + 5
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                pytest.fail("Core HTTP service did not start")
            time.sleep(.01)
        return url

    yield start
    for server, thread, sock in reversed(servers):
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        assert not thread.is_alive()


@pytest.mark.parametrize("depth,axis", [(1, 0), (2, 0), (1, -1)])
def test_http_predictions_match_direct_core_after_reopen(case_factory, serve, depth, axis):
    remote, direct = case_factory(depth, axis), case_factory(depth, axis)
    painting = remote.dataset.labels[:].copy()
    url = serve({"demo": remote.dataset})
    with httpx.Client(base_url=url, timeout=5) as client:
        assert client.get("/health").json() == {"status": "ok"}
        metadata = client.get("/datasets").json()["datasets"]["demo"]
        assert metadata["image"] == {"axes": ["c", "z", "y", "x"], "shape": [2, depth, 5, 4]}
        assert metadata["painting"] == {"axes": ["z", "y", "x"], "shape": [depth, 5, 4]}
        assert metadata["features"]["axis"] == axis
        assert metadata["features"]["axes"] == (["d", "z", "y", "x"] if axis == 0 else ["z", "y", "x", "d"])
        assert metadata["spatial_transform"] == {"scale": [3, 2, .5], "translation": [7, 11, 13],
                                                 "units": ["micrometer"] * 3}
        response = client.post("/datasets/demo/predict", json={})
        assert response.status_code == 409
        assert response.json()["code"] == "not_trained"
        assert client.post("/datasets/demo/train", json={}).status_code == 200
        response = client.post("/datasets/demo/predict", json={"batch_size": 7})
        assert response.status_code == 200
        assert response.json()["prediction_status"] == "ready"
        metadata = client.get("/datasets/demo").json()
        assert metadata["trained"] and not metadata["busy"]
        assert metadata["prediction"]["status"] == "ready"
    manager = default_manager(direct.dataset)
    manager.fit()
    manager.predict_to_segmentation(direct.dataset, batch_size=7)
    reopened = DataSet.from_paths(**remote.options)
    assert reopened.segmentation.metadata.zarr_format == 3
    np.testing.assert_array_equal(reopened.segmentation[:], direct.dataset.segmentation[:])
    np.testing.assert_array_equal(reopened.segmentation[:], remote.expected)
    np.testing.assert_array_equal(reopened.labels[:], painting)
    np.testing.assert_array_equal(reopened.image[:], remote.image)
    np.testing.assert_array_equal(zarr.open_array(remote.options["features_path"], mode="r")[:], remote.features)


def test_invalid_requests_do_not_start_operations(case_factory, serve):
    case = case_factory()
    with httpx.Client(base_url=serve({"demo": case.dataset}), timeout=5) as client:
        assert client.post("/datasets/missing/train").status_code == 404
        for value in [0, -1, True, 1.5, "7", None]:
            assert client.post("/datasets/demo/predict", json={"batch_size": value}).status_code == 400
        for data in ([], {"image_path": "elsewhere"}, {"batch_size": 7}):
            assert client.post("/datasets/demo/train", json=data).status_code == 400
        assert client.post("/datasets/demo/train", content="{").status_code == 400
        state = client.get("/datasets/demo").json()
        assert not state["trained"] and not state["busy"]
        assert state["prediction"]["status"] == "unknown"
        np.testing.assert_array_equal(case.dataset.segmentation[:], 0)


def test_training_failure_can_retry_and_invalidates_previous_model(case_factory, serve):
    case = case_factory(painted=False)
    with httpx.Client(base_url=serve({"demo": case.dataset}), timeout=5) as client:
        assert client.post("/datasets/demo/train").status_code == 500
        state = client.get("/datasets/demo").json()
        assert not state["trained"] and not state["busy"]
        case.dataset.labels[0, 2, 0] = 1
        case.dataset.labels[0, 2, 1] = 3
        assert client.post("/datasets/demo/train").status_code == 200
        assert client.post("/datasets/demo/predict").status_code == 200
        case.dataset.labels[:] = 0
        assert client.post("/datasets/demo/train").status_code == 500
        assert client.post("/datasets/demo/predict").json()["code"] == "not_trained"
        state = client.get("/datasets/demo").json()
        assert state["prediction"]["status"] == "unknown"
        assert not state["trained"] and not state["busy"]


def test_client_timeout_keeps_lock_until_core_finishes(case_factory, serve):
    first, second = case_factory(), case_factory()
    started, release = Event(), Event()

    class HeldManager:
        def __init__(self, dataset):
            self.delegate = default_manager(dataset)

        def fit(self):
            started.set()
            release.wait(10)
            self.delegate.fit()

    url = serve({"first": first.dataset, "second": second.dataset}, manager_factory=lambda dataset:
                HeldManager(dataset) if dataset is first.dataset else default_manager(dataset))
    with httpx.Client(base_url=url, timeout=3) as client:
        assert client.get("/health").status_code == 200
        try:
            with pytest.raises(httpx.ReadTimeout):
                client.post("/datasets/first/train", timeout=.2)
            assert started.wait(3)
            assert client.get("/datasets/first").json()["busy"]
            assert client.get("/health").status_code == 200
            response = client.post("/datasets/first/predict")
            assert response.status_code == 409 and response.json()["code"] == "busy"
            assert client.post("/datasets/second/train").status_code == 200
        finally:
            release.set()
        deadline = time.monotonic() + 5
        while True:
            state = client.get("/datasets/first").json()
            if not state["busy"]:
                assert state["trained"]
                break
            if time.monotonic() >= deadline:
                pytest.fail("Disconnected core operation did not complete")
            time.sleep(.01)


def test_partial_prediction_is_reported_and_retry_finishes(case_factory, serve):
    case = case_factory(depth=2)

    class FlakyTree:
        def __init__(self):
            self.tree = DecisionTreeClassifier(random_state=0)
            self.broken, self.calls = True, 0

        def fit(self, features, labels):
            return self.tree.fit(features, labels)

        def predict(self, rows):
            self.calls += 1
            if self.broken and self.calls == 2:
                raise OSError("prediction interrupted")
            return self.tree.predict(rows)

    model = FlakyTree()
    url = serve({"demo": case.dataset}, manager_factory=lambda dataset:
                SemanticSegmentationManager(DataManager([dataset]), model))
    with httpx.Client(base_url=url, timeout=5) as client:
        assert client.post("/datasets/demo/train").status_code == 200
        response = client.post("/datasets/demo/predict", json={"batch_size": 5})
        assert response.status_code == 500
        state = client.get("/datasets/demo").json()
        assert state["prediction"]["status"] == "failed"
        assert state["trained"] and not state["busy"]
        partial = case.dataset.segmentation[:]
        assert 0 < np.count_nonzero(partial) < partial.size
        model.broken = False
        assert client.post("/datasets/demo/predict", json={"batch_size": 5}).status_code == 200
        assert client.get("/datasets/demo").json()["prediction"]["status"] == "ready"
    np.testing.assert_array_equal(DataSet.from_paths(**case.options).segmentation[:], case.expected)


def test_service_imports_without_viewers_or_copick():
    code = """
import sys
from cellcanvas_service.app import create_app
for name in ('napari', 'qtpy', 'PySide6', 'PyQt5', 'copick', 'copick_server'):
    assert name not in sys.modules, name
print('headless import passed')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "headless import passed" in result.stdout


def test_cli_serves_configured_datasets(case_factory, tmp_path):
    case = case_factory()
    config = tmp_path / "service.json"
    config.write_text(json.dumps({"datasets": {"demo": case.options}}))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    with (tmp_path / "service.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "cellcanvas_service", str(config), "--port", str(port)],
                                   cwd=tmp_path, env=environment, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2) as client:
                while True:
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if process.poll() is not None or time.monotonic() >= deadline:
                        pytest.fail((tmp_path / "service.log").read_text())
                    time.sleep(.05)
                assert client.get("/datasets/demo").json()["image"]["shape"] == [2, 1, 5, 4]
                assert client.post("/datasets/demo/train").status_code == 200
                assert client.post("/datasets/demo/predict", json={"batch_size": 7}).status_code == 200
        finally:
            process.terminate()
            process.wait(timeout=10)
    np.testing.assert_array_equal(DataSet.from_paths(**case.options).segmentation[:], case.expected)
