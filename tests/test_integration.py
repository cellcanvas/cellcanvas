"""Service boundaries: declared labels, grid checks, inferred-background seeds, result records, durable restart,
provider registration and deployment profiles."""

import json
import os
from pathlib import Path
from threading import Event

import pytest

pytest.importorskip("cellcanvas_core", reason="Core service tests require the service extra")

import numpy as np
import zarr
from cellcanvas_core.data.data_set import DataSet
from starlette.testclient import TestClient

from cellcanvas_service.__main__ import load
from cellcanvas_service.app import create_app, label_digest
from cellcanvas_service.profiles import ProfileError, check, load_profile
from cellcanvas_service.providers import SEEDING_POLICY, Registry, fewshot_available, seeding_available
from cellcanvas_service.state import DirectoryStore

PROFILES = Path(__file__).parents[1] / "profiles"
SHAPE = (4, 12, 10)
SCALE, ORIGIN = [40.0, 40.0, 40.0], [0.0, 20.0, 0.0]
MANIFEST = {"model": "fixture features", "license": "CC0-1.0", "public": True, "source": "/local/source.zarr",
            "grid": {"shape": list(SHAPE), "axes": ["d", "z", "y", "x"], "resampling": None},
            "checkpoint": {"path": "/local/weights.pth", "sha256": "0" * 64}}
LABELS = {"classes": {"1": "background", "2": "lumen", "3": "membrane"}, "background": 1}
needs_seeding = pytest.mark.skipif(not seeding_available(), reason="cellcanvas-core without SeededDataManager (#20)")


def write_case(root, store=None):
    """Image, features and painting on a 4x12x10 grid; background (1) and membrane (3) painted, rest unknown."""
    rng = np.random.default_rng(0)
    z, y, x = np.indices(SHAPE)
    shell = (np.abs(y - 6) == 3).astype("f4")
    features = np.stack([shell + 0.05 * rng.standard_normal(SHAPE), 0.05 * rng.standard_normal(SHAPE),
                         z / 4]).astype("f4")
    location = (lambda name: f"{root}/{name}") if store is None else (lambda name: store(name))
    group = zarr.open_group(location("image.zarr"), mode="w", zarr_format=3)
    group.create_array("0", data=features[:1], dimension_names=("c", "z", "y", "x"))
    group.attrs["ome"] = {"version": "0.5", "multiscales": [{
        "axes": [{"name": "c", "type": "channel"}] + [
            {"name": a, "type": "space", "unit": "angstrom"} for a in ("z", "y", "x")],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1, 40, 40, 40]}, {"type": "translation", "translation": [0, 0, 20, 0]}]}],
    }]}
    zarr.save_array(location("features.zarr"), features, zarr_format=3)
    painting = zarr.open_array(location("painting.zarr"), mode="w", shape=SHAPE, dtype="i4", zarr_format=3,
                               dimension_names=("z", "y", "x"))
    labels = np.zeros(SHAPE, "i4")
    labels[1, 0, :3] = 1
    labels[2, 3, 2:5] = 3
    painting[...] = labels
    return {"image_path": location("image.zarr"), "features_path": location("features.zarr"),
            "labels_path": location("painting.zarr"), "segmentation_path": location("prediction.zarr"),
            "make_missing_datasets": True}, labels


def make_app(tmp_path, *, state=None, options=None, registry=None, name="case"):
    paths, labels = write_case(tmp_path / name) if not (tmp_path / name).exists() else (None, None)
    if paths is None:
        paths = json.loads((tmp_path / name / "paths.json").read_text())
        labels = np.asarray(zarr.open_array(paths["labels_path"])[...])
    else:
        (tmp_path / name / "paths.json").write_text(json.dumps(paths))
    dataset = DataSet.from_paths(**paths)
    defaults = {"features_manifest": MANIFEST, "labels": LABELS}
    app = create_app({"demo": dataset}, store=state, registry=registry,
                     dataset_options={"demo": defaults if options is None else options})
    return app, dataset, labels


def ready_receipt(client, path, body):
    from uuid import uuid4

    body = dict(body, operation_id=uuid4().hex, service_id=client.get("/providers").json()["service_id"])
    assert client.post(path, json=body).status_code == 202
    for _ in range(500):
        receipt = client.get(f"/datasets/demo/operations/{body['operation_id']}").json()
        if receipt["state"] != "running":
            return receipt
        Event().wait(.01)
    raise AssertionError("operation did not finish")


def test_dataset_description_reports_what_this_dataset_supports(tmp_path):
    app, _, _ = make_app(tmp_path)
    with TestClient(app) as client:
        metadata = client.get("/datasets/demo").json()
        providers = client.get("/providers").json()
    assert metadata["features_model"] == {"name": "fixture features", "public": True, "license": "CC0-1.0"}
    assert metadata["labels"] == {"unknown": 0, "background": 1, "classes": LABELS["classes"]}
    assert metadata["label_policies"] == ([SEEDING_POLICY] if seeding_available() else [])
    assert metadata["outputs"] == ["semantic"] and "rf" in metadata["classifiers"]
    # Prompting is another backend's job: the service lists prompt backends, it never claims to prompt.
    assert providers["prompt_backends"] == [] and "prompt" not in json.dumps(metadata["classifiers"])


def test_seeding_is_not_offered_without_a_declared_background_class(tmp_path):
    labels = {"classes": {"1": "cytosol", "2": "lumen", "3": "membrane"}}
    app, _, _ = make_app(tmp_path, options={"features_manifest": MANIFEST, "labels": labels})
    with TestClient(app) as client:
        assert client.get("/datasets/demo").json()["label_policies"] == []
        response = client.post("/datasets/demo/train", json={"label_policy": {"policy": SEEDING_POLICY}})
        assert response.status_code == 400
    with pytest.raises(ValueError, match="not available"):
        make_app(tmp_path, options={"labels": labels, "label_policy": {"policy": SEEDING_POLICY}})


@needs_seeding
def test_inferred_background_seeds_are_recorded_reproducible_and_never_painted(tmp_path):
    from cellcanvas_core.data.seeding import SeededDataManager

    app, dataset, labels = make_app(tmp_path)
    policy = {"policy": SEEDING_POLICY, "parameters": {"percentile": 30.0, "random_state": 7}}
    with TestClient(app) as client:
        receipt = ready_receipt(client, "/datasets/demo/train", {"label_policy": policy})
        assert receipt["state"] == "succeeded", receipt
        inferred = receipt["result"]["inferred"][0]
        record = client.get("/datasets/demo/model").json()
        seeds = client.get("/datasets/demo/inferred").json()
    # Human labels and unknown voxels are unchanged; seeds live only in the model provenance.
    np.testing.assert_array_equal(dataset.labels[...], labels)
    assert (np.asarray(dataset.labels[...]) == 0).sum() == (labels == 0).sum()
    assert inferred["policy"] == SEEDING_POLICY and inferred["label"] == 1 and inferred["count"] > 0
    assert inferred["parameters"] == {"percentile": 30.0, "random_state": 7, "background_label": 1}
    assert inferred["implementation"] == "cellcanvas_core.data.seeding.SeededDataManager"
    assert record["label_policy"] == {"policy": SEEDING_POLICY, "parameters": inferred["parameters"]}
    assert record["painting_digest"] == label_digest(labels)
    indices = np.asarray(seeds["indices"])
    assert seeds["inferred"]["indices_digest"] == inferred["indices_digest"] and len(indices) == inferred["count"]
    assert not labels.reshape(-1)[indices].any(), "seeds come only from unknown voxels"
    direct = SeededDataManager([dataset], **inferred["parameters"]).seed_indices(dataset)
    np.testing.assert_array_equal(np.sort(indices), np.sort(direct))


@needs_seeding
def test_label_policy_default_override_and_disable(tmp_path):
    options = {"features_manifest": MANIFEST, "labels": LABELS,
               "label_policy": {"policy": SEEDING_POLICY, "parameters": {"max_seeds": 3}}}
    app, _, _ = make_app(tmp_path, options=options)
    with TestClient(app) as client:
        assert client.get("/datasets/demo").json()["label_policy"]["policy"] == SEEDING_POLICY
        default = client.post("/datasets/demo/train", json={}).json()
        assert 0 < default["inferred"][0]["count"] <= 3
        disabled = client.post("/datasets/demo/train", json={"label_policy": None}).json()
        assert disabled["inferred"] == []
        assert client.get("/datasets/demo/inferred").status_code == 404
        override = {"policy": SEEDING_POLICY, "parameters": {"max_seeds": 1}}
        assert client.post("/datasets/demo/train", json={"label_policy": override}).json()["inferred"][0]["count"] == 1
        for bad in ({"background_label": 3}, {"max_seeds": 1.5}, {"percentile": True}, {"unknown": 1}):
            response = client.post("/datasets/demo/train", json={"label_policy": {"policy": SEEDING_POLICY,
                                                                                  "parameters": bad}})
            assert response.status_code == 400, bad
        assert client.post("/datasets/demo/train", json={"label_policy": {"policy": "other"}}).status_code == 400


def test_result_record_describes_the_complete_output(tmp_path):
    app, dataset, labels = make_app(tmp_path)
    with TestClient(app) as client:
        train = ready_receipt(client, "/datasets/demo/train", {})
        predict = ready_receipt(client, "/datasets/demo/predict", {"batch_size": 17})
        assert predict["state"] == "succeeded"
        body = client.get("/datasets/demo/results/1").json()
        assert client.get("/datasets/demo/results/2").status_code == 404
        response = client.get("/datasets/demo/prediction", params={"version": 1})
    record = body["result"]
    output = np.asarray(dataset.segmentation[...])
    assert record["digest"] == label_digest(output)
    assert body["current"] and record["operation_id"] == predict["id"] and record["version"] == 1
    assert record["painting_digest"] == label_digest(labels) and record["classifier"] == "RandomForestClassifier"
    assert record["scale_angstrom"] == SCALE and record["translation_angstrom"] == ORIGIN
    assert record["encoding"] == "semantic" and record["labels"]["classes"] == LABELS["classes"]
    assert record["features"]["checkpoint"] == {"sha256": "0" * 64}
    assert "/local" not in json.dumps(record), "result records carry no local paths"
    assert response.headers["X-Result-Digest"] == record["digest"]
    np.testing.assert_array_equal(np.frombuffer(response.content, output.dtype).reshape(SHAPE), output)
    assert train["result"]["inferred"] == []


def test_painting_refuses_undeclared_values_and_another_grid(tmp_path):
    app, dataset, _ = make_app(tmp_path)
    with TestClient(app) as client:
        service = client.get("/providers").json()["service_id"]
        body = {"service_id": service, "revision": "r1", "shape": list(SHAPE), "indices": [0], "values": [5]}
        assert client.put("/datasets/demo/painting", json=body).status_code == 400
        body.update(values=[2], scale=[10.0, 10.0, 10.0])
        response = client.put("/datasets/demo/painting", json=body)
        assert response.status_code == 409 and response.json()["code"] == "geometry_mismatch"
        body.update(scale=SCALE, translation=[0.0, 0.0, 0.0])
        assert client.put("/datasets/demo/painting", json=body).status_code == 409
        body.update(translation=ORIGIN)
        assert client.put("/datasets/demo/painting", json=body).status_code == 200
    assert dataset.labels[0, 0, 0] == 2 and int((np.asarray(dataset.labels[...]) != 0).sum()) == 1


def test_labels_keep_zero_as_unknown():
    from cellcanvas_service.app import Labels

    for bad in ({"classes": {"0": "background"}}, {"classes": {"1": "a"}, "background": 2}, {"classes": {}},
                {"classes": {"1": "a"}, "zero": "background"}):
        with pytest.raises(ValueError):
            Labels.from_config(bad)


def test_restart_keeps_receipts_model_and_completed_prediction(tmp_path):
    state = tmp_path / "state"
    app, _, _ = make_app(tmp_path, state=DirectoryStore(state))
    with TestClient(app) as client:
        service = client.get("/providers").json()["service_id"]
        body = {"service_id": service, "revision": "r1", "shape": list(SHAPE), "indices": [5, 6], "values": [1, 3]}
        assert client.put("/datasets/demo/painting", json=body).status_code == 200
        train = ready_receipt(client, "/datasets/demo/train", {})
        predict = ready_receipt(client, "/datasets/demo/predict", {})
        first = client.get("/datasets/demo/prediction", params={"version": 1}).content
    restarted, dataset, _ = make_app(tmp_path, state=DirectoryStore(state))
    with TestClient(restarted) as client:
        metadata = client.get("/datasets/demo").json()
        assert metadata["service_id"] == service and metadata["durable"]
        assert metadata["trained"] and metadata["model_painting_revision"] == "r1"
        assert metadata["painting"]["revision"] == "r1"
        assert metadata["prediction"] | {} == metadata["prediction"] and metadata["prediction"]["status"] == "ready"
        assert client.get(f"/datasets/demo/operations/{train['id']}").json()["state"] == "succeeded"
        assert client.get(f"/datasets/demo/operations/{predict['id']}").json()["state"] == "succeeded"
        assert client.get("/datasets/demo/prediction", params={"version": 1}).content == first
        assert client.get("/datasets/demo/results/1").json()["current"]
        # Predict again without retraining: the model was reloaded.
        assert client.post("/datasets/demo/predict", json={}).json()["prediction_version"] == 2
        # An out-of-band change to the stored output is detected, not served.
        dataset.segmentation[0, 0, 0] = 9
        assert client.get("/datasets/demo/prediction", params={"version": 2}).status_code == 409


def test_restart_reports_interrupted_work_and_refuses_partial_prediction(tmp_path):
    state = tmp_path / "state"
    started, release = Event(), Event()

    class Blocking:
        def __init__(self, model):
            self.model = model

        def fit(self):
            started.set()
            release.wait(10)

    first, dataset, _ = make_app(tmp_path, state=DirectoryStore(state))
    try:
        with TestClient(first) as client:
            ready_receipt(client, "/datasets/demo/train", {})
            assert ready_receipt(client, "/datasets/demo/predict", {})["state"] == "succeeded"
            # Simulate a process that stopped while writing the next prediction and while training.
            store = DirectoryStore(state)
            store.save_record("prediction", "demo", {"status": "writing", "version": 2, "operation_id": "x"})
            dataset.segmentation[0, 0, 0] = 2
            first.state.datasets["demo"].manager = Blocking(first.state.datasets["demo"].manager.model)
            from uuid import uuid4

            identifier = uuid4().hex
            service = client.get("/providers").json()["service_id"]
            assert client.post("/datasets/demo/train", json={"operation_id": identifier,
                                                            "service_id": service}).status_code == 202
            assert started.wait(5)
            second, _, _ = make_app(tmp_path, state=DirectoryStore(state))
            with TestClient(second) as restarted:
                receipt = restarted.get(f"/datasets/demo/operations/{identifier}").json()
                assert receipt["state"] == "failed" and receipt["error"]["code"] == "interrupted"
                metadata = restarted.get("/datasets/demo").json()
                assert metadata["prediction"]["status"] == "failed"
                assert restarted.get("/datasets/demo/prediction", params={"version": 1}).status_code == 409
                assert restarted.get("/datasets/demo/prediction", params={"version": 2}).status_code == 409
                retry = restarted.post("/datasets/demo/predict", json={}).json()
                assert retry["prediction_status"] == "ready" and retry["prediction_version"] == 2
    finally:
        release.set()


def test_classifier_registry_reports_only_installed_classifiers(tmp_path):
    app, _, _ = make_app(tmp_path)
    with TestClient(app) as client:
        names = [c["name"] for c in client.get("/providers").json()["classifiers"]]
        assert "rf" in names and ("fewshot" in names) == fewshot_available()
        assert client.post("/datasets/demo/train", json={"classifier": "missing"}).status_code == 400
        result = client.post("/datasets/demo/train", json={"classifier": "rf"}).json()
        assert result["trained"]
        assert client.get("/datasets/demo/model").json()["classifier"] == "rf"
        if fewshot_available():
            assert client.post("/datasets/demo/train", json={"classifier": "fewshot"}).json()["trained"]
            predicted = client.post("/datasets/demo/predict", json={}).json()
            assert predicted["prediction_status"] == "ready"
            values = set(np.unique(np.asarray(app.state.datasets["demo"].dataset.segmentation[...])).tolist())
            assert values <= {1, 3}, "few-shot predictions use the painted class values"


class PrivateProvider:
    name = "private-fixture"
    public = False

    def check_manifest(self, manifest):
        if manifest.get("provider") != self.name:
            raise ValueError("not produced by this provider")


def test_private_features_need_an_allowed_registered_provider(tmp_path):
    private = dict(MANIFEST, public=False, license="proprietary", provider="private-fixture")
    options = {"features_manifest": private, "features_provider": "private-fixture", "labels": LABELS}
    with pytest.raises(ValueError, match="does not allow private providers"):
        make_app(tmp_path, options=options)
    with pytest.raises(ValueError, match="not registered"):
        make_app(tmp_path, options=options, registry=Registry(allow_private=True))
    registry = Registry(allow_private=True, feature_providers={"private-fixture": PrivateProvider()})
    app, _, _ = make_app(tmp_path, options=options, registry=registry)
    with TestClient(app) as client:
        assert client.get("/datasets/demo").json()["features_model"]["public"] is False
        ready_receipt(client, "/datasets/demo/train", {})
        ready_receipt(client, "/datasets/demo/predict", {})
        record = client.get("/datasets/demo/results/1").json()["result"]
    assert record["features"]["public"] is False and record["features"]["provider"] == "private-fixture"


ENV = {"CELLCANVAS_SERVICE_CONFIG": "service.json", "CELLCANVAS_STATE_DIR": "/srv/state",
       "CELLCANVAS_DATA_ROOT": "/srv/data"}


@pytest.mark.parametrize("name", ["local", "reef", "bruno", "aws"])
def test_deployment_profiles_validate(name):
    environ = dict(ENV, CELLCANVAS_DATA_ROOT="s3://bucket/project" if name == "aws" else "/srv/data")
    profile = load_profile(PROFILES / f"{name}.json", environ)
    assert profile["name"] == name and isinstance(profile["bind"]["port"], int)
    assert profile["providers"]["allow_private"] is False
    assert profile["data"]["kind"] == ("fsspec" if name == "aws" else "posix")
    assert (profile["bind"]["host"] == "127.0.0.1") == (profile["bind"]["exposure"] == "loopback")


def test_profiles_refuse_missing_settings_credentials_and_open_binds(tmp_path):
    with pytest.raises(ProfileError, match="CELLCANVAS_DATA_ROOT"):
        load_profile(PROFILES / "reef.json", {"CELLCANVAS_SERVICE_CONFIG": "x", "CELLCANVAS_STATE_DIR": "y"})
    raw = json.loads((PROFILES / "aws.json").read_text())
    raw["data"]["storage_options"] = {"client_kwargs": {"aws_secret_access_key": "literal"}}
    (tmp_path / "secret.json").write_text(json.dumps(raw))
    with pytest.raises(ProfileError, match="credentials"):
        load_profile(tmp_path / "secret.json", dict(ENV))
    raw = json.loads((PROFILES / "local.json").read_text())
    raw["bind"]["host"] = "0.0.0.0"
    (tmp_path / "open.json").write_text(json.dumps(raw))
    with pytest.raises(ProfileError, match="behind_proxy"):
        load_profile(tmp_path / "open.json", {})


def test_object_store_profile_trains_and_predicts_from_fsspec_urls(tmp_path, monkeypatch):
    """The AWS profile path, replayed against an in-memory fsspec filesystem instead of S3."""
    fsspec = pytest.importorskip("fsspec")
    from fsspec.implementations.memory import MemoryFileSystem
    from zarr.storage import FsspecStore

    from cellcanvas_service.profiles import core_accepts_storage_options

    if not core_accepts_storage_options():
        pytest.skip("cellcanvas-core without DataSet.from_paths(storage_options)")

    class RemoteMemory(MemoryFileSystem):
        protocol = ("cellcanvasremote",)
        store = {}
        pseudo_dirs = [""]

    fsspec.register_implementation("cellcanvasremote", RemoteMemory, clobber=True)
    root = f"cellcanvasremote://cellcanvas-{os.getpid()}-{tmp_path.name}"
    write_case(None, store=lambda name: FsspecStore.from_url(f"{root}/{name}", read_only=False))
    config = {"datasets": {"demo": {"image_path": "image.zarr", "features_path": "features.zarr",
                                    "labels_path": "painting.zarr", "segmentation_path": "prediction.zarr",
                                    "make_missing_datasets": True, "features_manifest": MANIFEST,
                                    "labels": LABELS}}}
    (tmp_path / "service.json").write_text(json.dumps(config))
    environ = dict(ENV, CELLCANVAS_DATA_ROOT=root, CELLCANVAS_STATE_DIR=str(tmp_path / "state"),
                   CELLCANVAS_SERVICE_CONFIG=str(tmp_path / "service.json"))
    profile = load_profile(PROFILES / "aws.json", environ)
    assert check(PROFILES / "aws.json", environ=environ)["datasets"]["demo"]["image_path"] == f"{root}/image.zarr"
    app = load(profile["config"], profile)
    with TestClient(app) as client:
        assert client.get("/datasets/demo").json()["storage"]["image"] is None
        ready_receipt(client, "/datasets/demo/train", {})
        assert ready_receipt(client, "/datasets/demo/predict", {})["state"] == "succeeded"
        record = client.get("/datasets/demo/results/1").json()["result"]
    stored = zarr.open_array(FsspecStore.from_url(f"{root}/prediction.zarr", read_only=True))[...]
    assert record["digest"] == label_digest(np.asarray(stored))
    assert (tmp_path / "state" / "results" / "demo" / "1.json").exists()
