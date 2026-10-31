"""Exercise the legacy Copick HTTP protocol against real project storage."""

import json
import runpy
import socket
import sys
import time
from pathlib import Path
from threading import Event, Thread, get_ident

import pytest

copick = pytest.importorskip("copick", reason="Legacy workflow tests require the server/client extras")
import numpy as np
import requests
import uvicorn
import zarr
from fsspec import get_mapper
from starlette.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def server_module():
    return runpy.run_path(str(ROOT / "workflows/cellcanvas-server/server.py"))


@pytest.fixture(scope="module")
def client_module():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "workflows/napari-client"))
        yield runpy.run_path(str(ROOT / "workflows/napari-client/client.py"))


@pytest.fixture(params=[1, 2])
def project(tmp_path, request):
    config = tmp_path / "project.json"
    config.write_text(json.dumps({
        "config_type": "filesystem", "name": "HTTP test", "description": "synthetic",
        "version": "1.0.0", "pickable_objects": [],
        "overlay_root": str(tmp_path / "overlay"), "overlay_fs_args": {"auto_mkdir": True},
    }))
    root = copick.from_file(str(config))
    run = root.new_run("run-one")
    spacing = run.new_voxel_spacing(2.0)
    tomogram = spacing.new_tomogram("wbp")
    values = np.arange(request.param * 5 * 4, dtype="f4").reshape(request.param, 5, 4)
    group = zarr.group(store=tomogram.zarr(), overwrite=True)
    group.create_dataset("0", data=values)
    return root, config, values


@pytest.fixture
def request_log():
    return []


@pytest.fixture
def live_server(project, server_module, request_log):
    root, _, _ = project
    app = server_module["create_cellcanvas_copick_app"](root)

    async def record_request(request, call_next):
        request_log.append((request.method, request.url.path))
        return await call_next(request)

    app.add_middleware(BaseHTTPMiddleware, dispatch=record_request)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    address = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                pytest.fail("HTTP server did not start")
            time.sleep(.01)
        yield address
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        assert not thread.is_alive()


def test_http_painting_and_remote_reopen(live_server, project, client_module):
    root, config, values = project
    client = client_module["CellCanvasClient"](live_server)
    assert client.test_connection()
    context = {"run_name": "run-one", "voxel_size": 2.0, "tomogram_type": "wbp",
               "user_id": "napariUser", "session_id": "17"}
    created = client.send_request("api/segmentation/ensure", context, "POST")
    assert created["status"] == "success"
    assert created["segmentations"]["painting"]["created"]
    coordinates = [[0, 2, 0], [values.shape[0] - 1, 2, 1]]
    result = client.send_request("api/painting/update", dict(context, coordinates=coordinates, label=3), "POST")
    assert result == {"status": "success", "updates": 2, "errors": 0, "label": 3}
    # Erase uses the same explicit zero label as local painting.
    result = client.send_request("api/painting/update", dict(context, coordinates=[coordinates[0]], label=0), "POST")
    assert result["updates"] == 1
    expected = np.zeros(values.shape, dtype="i4")
    expected[tuple(coordinates[1])] = 3
    path = "/run-one/Segmentations/2.0_napariUser_17_painting-multilabel.zarr"
    remote = zarr.open_group(get_mapper(live_server + path), mode="r")
    np.testing.assert_array_equal(remote["data"][:], expected)
    # Check both HTTP methods used by remote Zarr stores, including missing keys.
    for method in (requests.get, requests.head):
        assert method(live_server + path + "/data/.zarray", timeout=2).status_code == 200
        assert method(live_server + path + "/missing", timeout=2).status_code == 404
    repeated = client.send_request("api/segmentation/ensure", context, "POST")
    assert not repeated["segmentations"]["painting"]["created"]
    reopened = copick.from_file(str(config)).get_run("run-one")
    painting = reopened.get_segmentations(voxel_size=2.0, name="painting", user_id="napariUser",
                                          session_id="17", is_multilabel=True)[0]
    np.testing.assert_array_equal(zarr.open_group(painting.zarr(), mode="r")["data"][:], expected)
    tomo = reopened.get_voxel_spacing(2.0).get_tomogram("wbp")
    np.testing.assert_array_equal(zarr.open_group(tomo.zarr(), mode="r")["0"][:], values)
    remote_tomo = zarr.open_group(get_mapper(live_server + "/run-one/Tomograms/VoxelSpacing2.0/wbp.zarr"), mode="r")
    np.testing.assert_array_equal(remote_tomo["0"][:], values)


def test_optional_superpixels_do_not_block_server(server_module, project, monkeypatch):
    monkeypatch.setitem(sys.modules, "cellcanvas_spp.segmentation", None)
    app = server_module["create_cellcanvas_copick_app"](project[0])
    with TestClient(app) as client:
        assert client.get("/ping").status_code == 200
        response = client.post("/api/superpixels/create", json={})
        assert response.status_code == 503
        assert "cellcanvas-spp" in response.json()["error"]
        assert client.get("/unknown/path/key").status_code == 404


@pytest.fixture(scope="session")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def wait_for(app):
    def wait(predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate():
            app.processEvents()
            if time.monotonic() >= deadline:
                pytest.fail("Painting synchronization timed out")
            time.sleep(.001)
        app.processEvents()
    return wait


def test_client_widget_startup_and_worker_cleanup(client_module, app):
    from napari.components import ViewerModel
    from qtpy.QtCore import QEvent
    from qtpy.QtWidgets import QApplication
    widget = client_module["CopickPlugin"](ViewerModel())
    try:
        widget.close()
        assert widget.painting_sync is None
        assert not widget.cellcanvas_client.connected
    finally:
        widget.cleanup()
        widget.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()


@pytest.fixture
def painting_client(client_module, live_server, project, app, wait_for):
    from napari.components import ViewerModel
    from qtpy.QtCore import QEvent
    from qtpy.QtWidgets import QApplication
    widget = client_module["CopickPlugin"](ViewerModel())
    widget.root = project[0]
    widget.cellcanvas_client = client_module["CellCanvasClient"](live_server)
    assert widget.ensure_and_load_segmentations("run-one", 2.0, "wbp")
    assert "Connected to" in widget.cellcanvas_status_label.text()
    yield widget
    if widget.painting_sync is not None and widget.painting_sync.failed:
        widget.painting_sync.retry()
    wait_for(lambda: widget.painting_sync is None or not widget.painting_sync.pending)
    widget.close()
    widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def saved_painting(root, run="run-one"):
    segmentation = root.get_run(run).get_segmentations(voxel_size=2.0, name="painting", user_id="napariUser",
                                                       session_id="17", is_multilabel=True)[0]
    return zarr.open_group(segmentation.zarr(), mode="r")["data"][:]


def test_gui_brush_erase_undo_redo_and_reopen(painting_client, project, wait_for, request_log):
    widget = painting_client
    layer = widget.painting_layer
    assert isinstance(layer.data, np.ndarray)
    assert not widget.prediction_layer.editable
    layer.brush_size = 3
    # The saved value must come from the data, not the selected tool label.
    layer.selected_label = 9
    layer.paint((0, 2, 1), 3)
    wait_for(lambda: not widget.painting_sync.pending)
    expected = layer.data.copy()
    assert np.count_nonzero(expected) > 1
    np.testing.assert_array_equal(saved_painting(project[0]), expected)
    layer.paint((0, 2, 1), 0)
    wait_for(lambda: not widget.painting_sync.pending)
    np.testing.assert_array_equal(saved_painting(project[0]), layer.data)
    layer.undo()
    wait_for(lambda: not widget.painting_sync.pending)
    np.testing.assert_array_equal(saved_painting(project[0]), expected)
    layer.redo()
    wait_for(lambda: not widget.painting_sync.pending)
    np.testing.assert_array_equal(saved_painting(project[0]), layer.data)
    reopened = copick.from_file(str(project[1]))
    np.testing.assert_array_equal(saved_painting(reopened), layer.data)
    assert widget.painting_status_label.text() == "Painting saved."
    assert not any(method == "PUT" for method, path in request_log)
    assert ("POST", "/api/painting/update") in request_log
    source = project[0].get_run("run-one").get_voxel_spacing(2.0).get_tomogram("wbp")
    np.testing.assert_array_equal(zarr.open_group(source.zarr(), mode="r")["0"][:], project[2])


def test_failed_http_save_retains_edits_and_retries(painting_client, project, wait_for, monkeypatch):
    widget = painting_client
    root = project[0]
    layer = widget.painting_layer
    with monkeypatch.context() as patch:
        def fail_lookup(self, name):
            raise OSError("temporary storage fault")
        patch.setattr(type(root), "get_run", fail_lookup)
        layer.brush_size = 1
        layer.paint((0, 2, 1), 3)
        wait_for(lambda: widget.painting_sync.failed and widget.painting_sync._worker is None)
        assert layer.data[0, 2, 1] == 3
        assert widget.retry_painting_button.isEnabled()
        assert "not saved" in widget.painting_status_label.text()
        before = len(widget.viewer.layers)
        assert not widget.ensure_and_load_segmentations("other-run", 2.0, "wbp")
        assert not widget.load_config(project[1])
        assert widget.load_tomogram(None) is None
        assert len(widget.viewer.layers) == before
        assert widget.close() is False
        layer.paint((0, 2, 1), 0)
        layer.paint((0, 2, 2), 7)
    widget.retry_painting_button.click()
    wait_for(lambda: not widget.painting_sync.pending)
    np.testing.assert_array_equal(saved_painting(root), layer.data)
    assert saved_painting(root)[0, 2, 2] == 7
    assert not widget.retry_painting_button.isEnabled()


def test_slow_save_is_responsive_and_coalesces_later_edits(painting_client, project, wait_for, monkeypatch):
    from qtpy.QtCore import QTimer
    widget = painting_client
    sync, layer = widget.painting_sync, widget.painting_layer
    started, release = Event(), Event()
    calls, ticks = [], []
    original = sync.client.send_request

    def slow(endpoint, params=None, method="GET"):
        calls.append((get_ident(), params["run_name"], params["label"]))
        started.set()
        release.wait(5)
        return original(endpoint, params, method)

    monkeypatch.setattr(sync.client, "send_request", slow)
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start(5)
    try:
        layer.brush_size = 1
        layer.paint((0, 2, 1), 3)
        sync.flush()
        wait_for(lambda: started.is_set() and len(ticks) >= 5)
        assert calls[0][0] != get_ident()
        assert widget.close() is False
        assert not widget.ensure_and_load_segmentations("other-run", 2.0, "wbp")
        layer.paint((0, 2, 1), 7)
        layer.paint((0, 2, 2), 9)
    finally:
        release.set()
        timer.stop()
    wait_for(lambda: not sync.pending)
    np.testing.assert_array_equal(saved_painting(project[0]), layer.data)
    assert saved_painting(project[0])[0, 2, 1] == 7


def test_switch_disconnects_old_buffer_and_captures_new_context(painting_client, project, wait_for):
    widget = painting_client
    root = project[0]
    old_layer, old_sync = widget.painting_layer, widget.painting_sync
    old_layer.brush_size = 1
    old_layer.paint((0, 2, 1), 3)
    wait_for(lambda: not old_sync.pending)
    second_run = root.new_run("run-two")
    tomo = second_run.new_voxel_spacing(2.0).new_tomogram("wbp")
    zarr.group(tomo.zarr(), overwrite=True).create_dataset("0", data=project[2])
    assert widget.ensure_and_load_segmentations("run-two", 2.0, "wbp")
    assert not old_layer.editable
    old_layer.paint((0, 2, 1), 9)
    widget.current_run_name = "wrong-after-binding"
    widget.session_id = "different-after-binding"
    new_layer = widget.painting_layer
    new_layer.brush_size = 1
    new_layer.paint((0, 2, 2), 7)
    wait_for(lambda: not widget.painting_sync.pending)
    assert saved_painting(root, "run-one")[0, 2, 1] == 3
    assert saved_painting(root, "run-two")[0, 2, 2] == 7
    assert saved_painting(root, "run-two")[0, 2, 1] == 0


def test_value_changes_preserve_mixed_labels_and_noop_does_not_post(painting_client, project, wait_for, request_log):
    widget = painting_client
    layer = widget.painting_layer
    indices = (np.array([0, 0]), np.array([2, 2]), np.array([1, 2]))
    layer.data_setitem(indices, np.array([3, 7]))
    wait_for(lambda: not widget.painting_sync.pending)
    np.testing.assert_array_equal(saved_painting(project[0]), layer.data)
    layer.undo()
    wait_for(lambda: not widget.painting_sync.pending)
    np.testing.assert_array_equal(saved_painting(project[0]), 0)
    layer.redo()
    wait_for(lambda: not widget.painting_sync.pending)
    np.testing.assert_array_equal(saved_painting(project[0]), layer.data)
    count = sum(method == "POST" for method, path in request_log)
    layer.refresh()
    wait_for(lambda: not widget.painting_sync.pending)
    assert sum(method == "POST" for method, path in request_log) == count


def test_lost_acknowledgment_retry_clears_reverted_voxels(painting_client, project, wait_for, monkeypatch):
    widget = painting_client
    sync, layer = widget.painting_sync, widget.painting_layer
    original = sync.client.send_request

    def lose_reply(endpoint, params=None, method="GET"):
        response = original(endpoint, params, method)
        assert response["status"] == "success"
        sync.client.last_error = "reply lost after server write"
        return None

    with monkeypatch.context() as patch:
        patch.setattr(sync.client, "send_request", lose_reply)
        layer.brush_size = 1
        layer.paint((0, 2, 1), 3)
        wait_for(lambda: sync.failed and sync._worker is None)
        assert saved_painting(project[0])[0, 2, 1] == 3
        layer.undo()
        assert layer.data[0, 2, 1] == 0
    widget.retry_painting_button.click()
    wait_for(lambda: not sync.pending)
    np.testing.assert_array_equal(saved_painting(project[0]), layer.data)
    assert saved_painting(project[0])[0, 2, 1] == 0


def test_window_close_waits_for_saved_painting(painting_client, wait_for, app):
    from qtpy.QtCore import QEvent
    from qtpy.QtWidgets import QApplication, QMainWindow
    widget = painting_client
    window = QMainWindow()
    window.setCentralWidget(widget)
    window.installEventFilter(widget)
    try:
        widget.painting_layer.brush_size = 1
        widget.painting_layer.paint((0, 2, 1), 3)
        assert window.close() is False
        wait_for(lambda: not widget.painting_sync.pending)
        assert window.close() is True
    finally:
        window.takeCentralWidget()
        widget.setParent(None)
        window.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_batching_preserves_values_and_rejects_partial_ack(client_module):
    from cellcanvas_painting import save_changes
    previous = np.zeros((1, 1, 8), dtype="i4")
    current = previous.copy()
    current[0, 0, 0] = 3
    current[0, 0, 7] = 7
    uncertain = np.zeros_like(current, dtype=bool)
    uncertain[0, 0, 2] = True
    calls = []

    class Client:
        last_error = None

        def send_request(self, endpoint, params, method):
            calls.append(params)
            return {"status": "success", "updates": len(params["coordinates"]), "errors": 0}

    result = save_changes(Client(), {"run_name": "frozen"}, previous, current, uncertain, batch_size=3)
    np.testing.assert_array_equal(result, current)
    reconstructed = previous.copy()
    for payload in calls:
        assert len(payload["coordinates"]) <= 3
        assert payload["run_name"] == "frozen"
        reconstructed[tuple(np.array(payload["coordinates"]).T)] = payload["label"]
    np.testing.assert_array_equal(reconstructed, current)
    assert any(payload["label"] == 0 for payload in calls)

    class PartialClient:
        last_error = None

        def send_request(self, endpoint, params, method):
            return {"status": "success", "updates": len(params["coordinates"]), "errors": 1}

    with pytest.raises(RuntimeError, match="confirm every"):
        save_changes(PartialClient(), {}, previous, current)


def test_offline_edits_can_be_backed_up_before_closing(painting_client, project, wait_for, monkeypatch, client_module, tmp_path):
    widget = painting_client
    root = project[0]
    with monkeypatch.context() as patch:
        def fail_lookup(self, name):
            raise OSError("offline")
        patch.setattr(type(root), "get_run", fail_lookup)
        widget.painting_layer.brush_size = 1
        widget.painting_layer.paint((0, 2, 1), 3)
        wait_for(lambda: widget.painting_sync.failed and widget.painting_sync._worker is None)
    expected = widget.painting_layer.data.copy()
    backup = tmp_path / "painting.npy"
    dialog = client_module["QFileDialog"]
    monkeypatch.setattr(dialog, "getSaveFileName", lambda *args: (str(tmp_path / "missing/painting.npy"), ""))
    widget.save_painting_copy_button.click()
    assert widget.painting_sync.failed
    assert "Could not save" in widget.info_label.text()
    monkeypatch.setattr(dialog, "getSaveFileName", lambda *args: (str(backup), ""))
    widget.save_painting_copy_button.click()
    np.testing.assert_array_equal(np.load(backup, allow_pickle=False), expected)
    assert widget.painting_sync is None
    assert not widget.painting_layer.editable
    assert widget.close() is True
    np.testing.assert_array_equal(saved_painting(root), 0)
