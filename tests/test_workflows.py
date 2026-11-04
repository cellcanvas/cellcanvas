"""Exercise the legacy Copick HTTP protocol against real project storage."""

import json
import queue
import runpy
import socket
import sys
import time
from pathlib import Path
from threading import Thread

import copick
import numpy as np
import pytest
import requests
import uvicorn
import zarr
from fsspec import get_mapper
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def server_module():
    return runpy.run_path(str(ROOT / "workflows/cellcanvas-server/server.py"))


@pytest.fixture(scope="module")
def client_module():
    return runpy.run_path(str(ROOT / "workflows/napari-client/client.py"))


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
def live_server(project, server_module):
    root, _, _ = project
    app = server_module["create_cellcanvas_copick_app"](root)
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


def test_client_widget_startup_and_worker_cleanup(client_module):
    from napari.components import ViewerModel
    from qtpy.QtCore import QEvent
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    widget = client_module["CopickPlugin"](ViewerModel())
    updates = queue.Queue()
    thread = Thread(target=widget.process_update_queue, daemon=True)
    widget.update_queue = updates
    try:
        thread.start()
        widget.close()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert not widget.cellcanvas_client.connected
    finally:
        widget.cleanup()
        thread.join(timeout=2)
        widget.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
