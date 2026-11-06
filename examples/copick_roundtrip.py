"""Train CellCanvas on a copick painting and publish the completed prediction back into copick.

    python examples/copick_roundtrip.py COPICK_CONFIG RUN VOXEL_SIZE TOMO_TYPE \
        --service http://127.0.0.1:8001 --dataset NAME \
        --painting NAME:USER:SESSION --output NAME:USER:SESSION

This runs in a copick environment (Zarr 2) and talks to ``python -m cellcanvas_service`` (Zarr 3) over HTTP:

1. open the reconstruction and its multilabel painting in copick (0 is unknown, positive values are classes);
2. upload the painting with the reconstruction's voxel grid; the service refuses another grid or undeclared classes;
3. train, then predict, as receipted operations (a service restart reports interrupted work, never success);
4. read the completed prediction by version and check its digest against the service's result record;
5. write it into a separate copick segmentation, never into the painting.

Only the standard library, numpy, zarr and copick are needed on this side.
"""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
import uuid

import copick
import numpy as np
import zarr


def digest(array):
    """The service's label digest: sha256 of dtype, shape and C-order bytes."""
    array = np.ascontiguousarray(array)
    header = f"{array.dtype.str}:{','.join(map(str, array.shape))}:".encode()
    return "sha256:" + hashlib.sha256(header + array.tobytes()).hexdigest()


class Service:
    def __init__(self, url, dataset):
        self.base = f"{url.rstrip('/')}/datasets/{dataset}"

    def send(self, method, path="", body=None):
        """Response bytes and headers; service errors are raised with their code."""
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request) as response:
                return response.read(), response.headers
        except urllib.error.HTTPError as error:
            detail = json.loads(error.read() or b"{}")
            raise RuntimeError(f"{method} {path or '/'}: {detail.get('code')}: {detail.get('error')}") from None

    def call(self, method, path="", body=None) -> dict:
        return json.loads(self.send(method, path, body)[0])

    def run(self, operation, service_id, timeout=3600):
        """Submit a receipted operation and wait for its final state."""
        receipt = self.call("POST", f"/{operation}", {"operation_id": uuid.uuid4().hex, "service_id": service_id})
        deadline = time.time() + timeout
        while receipt["state"] == "running" and time.time() < deadline:
            time.sleep(1)
            receipt = self.call("GET", f"/operations/{receipt['id']}")
        if receipt["state"] != "succeeded":
            raise RuntimeError(f"{operation} {receipt['state']}: {receipt.get('error')}")
        return receipt["result"]


def grid(tomogram):
    """Level-0 shape, scale and translation (angstrom, z/y/x) of a copick tomogram."""
    group = zarr.open_group(tomogram.zarr(), mode="r")
    transforms = group.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    scale = next(t["scale"] for t in transforms if t["type"] == "scale")
    translation = next((t["translation"] for t in transforms if t["type"] == "translation"), [0.0, 0.0, 0.0])
    return list(group["0"].shape), [float(v) for v in scale], [float(v) for v in translation]


def segmentation(run, voxel_size, ref):
    name, user, session = ref
    found = run.get_segmentations(name=name, user_id=user, session_id=session, voxel_size=voxel_size,
                                  is_multilabel=True)
    return found[0] if found else None


def roundtrip(config, run_name, voxel_size, tomo_type, service_url, dataset, painting_ref, output_ref):
    root = copick.from_file(config)
    run = root.get_run(run_name)
    tomogram = run.get_voxel_spacing(voxel_size).get_tomogram(tomo_type)
    shape, scale, translation = grid(tomogram)
    painting = segmentation(run, voxel_size, painting_ref)
    if painting is None:
        raise SystemExit("No painting segmentation to train on.")
    labels = np.asarray(painting.numpy())
    service = Service(service_url, dataset)
    described = service.call("GET")
    service_id = described["service_id"]

    flat = labels.reshape(-1)
    indices = np.flatnonzero(flat)
    revision = "content-" + digest(labels)[7:39]
    service.call("PUT", "/painting", {"service_id": service_id, "revision": revision, "shape": shape,
                                      "scale": scale, "translation": translation,
                                      "indices": indices.tolist(), "values": flat[indices].tolist()})
    trained = service.run("train", service_id)
    predicted = service.run("predict", service_id)
    version = predicted["prediction_version"]
    body, headers = service.send("GET", f"/prediction?version={version}")
    record = service.call("GET", f"/results/{version}")["result"]
    output = np.frombuffer(body, dtype=headers["X-Dtype"]).reshape(shape)
    if digest(output) != record["digest"] or record["painting_revision"] != revision:
        raise SystemExit("The prediction read does not match its result record; not publishing.")

    if output_ref == painting_ref:
        raise SystemExit("Refusing to publish predictions into the painting they were trained on.")
    name, user, session = output_ref
    target = segmentation(run, voxel_size, output_ref) or run.new_segmentation(
        voxel_size=voxel_size, name=name, session_id=session, is_multilabel=True, user_id=user)
    target.from_numpy(output.astype(np.uint8), dtype=np.uint8)
    return {"painting_revision": revision, "painted_voxels": int(indices.size),
            "inferred": trained["inferred"], "prediction_version": version, "digest": record["digest"],
            "classes": {str(v): int(c) for v, c in zip(*np.unique(output, return_counts=True))},
            "features": record["features"]["model"] if record["features"] else None}


def reference(text):
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("use NAME:USER:SESSION")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("run")
    parser.add_argument("voxel_size", type=float)
    parser.add_argument("tomo_type")
    parser.add_argument("--service", default="http://127.0.0.1:8001")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--painting", type=reference, default=("cellcanvas-painting", "cellcanvas", "1"))
    parser.add_argument("--output", type=reference, default=("cellcanvas-prediction", "cellcanvas", "1"))
    args = parser.parse_args()
    print(json.dumps(roundtrip(args.config, args.run, args.voxel_size, args.tomo_type, args.service, args.dataset,
                               args.painting, args.output), indent=2))


if __name__ == "__main__":
    main()
