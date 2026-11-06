"""Durable operation receipts, trained models and published-result records.

With a state directory, a restarted service keeps its ``service_id``, answers for every operation it accepted, and
reloads the last trained model of each dataset. Work that was running when the process stopped is reported as
failed with code ``interrupted``; it is never reported as succeeded. A prediction whose write was interrupted, or
whose stored voxels no longer match the recorded digest, is not served.

Every record is written to a temporary file, flushed and renamed, so a crash leaves either the previous or the new
record. Models are pickles the service wrote itself; their SHA-256 is recorded and checked before loading. Do not
point a state directory at files from another source.

Without a state directory the service keeps the same records in memory, for tests and throwaway sessions.
"""

import hashlib
import json
import os
import pickle
from pathlib import Path
from uuid import uuid4


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with open(temporary, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class MemoryStore:
    durable = False

    def __init__(self):
        self.service_id = uuid4().hex
        self._records = {}
        self._models = {}

    def save_operation(self, dataset, job):
        self._records[("operation", dataset, job["id"])] = json.loads(json.dumps(job))

    def operations(self, dataset):
        return [dict(v) for (kind, name, _), v in self._records.items() if kind == "operation" and name == dataset]

    def save_record(self, kind, dataset, record):
        self._records[(kind, dataset, None)] = json.loads(json.dumps(record))

    def record(self, kind, dataset):
        value = self._records.get((kind, dataset, None))
        return None if value is None else dict(value)

    def save_result(self, dataset, version, manifest):
        self._records[("result", dataset, version)] = manifest

    def result(self, dataset, version):
        return self._records.get(("result", dataset, version))

    def save_model(self, dataset, model, record):
        self._models[dataset] = model
        self.save_record("model", dataset, record)

    def load_model(self, dataset):
        record = self.record("model", dataset)
        return (self._models[dataset], record) if record and dataset in self._models else (None, None)


class DirectoryStore(MemoryStore):
    """Records under one directory: service.json, operations/, models/, predictions/ and results/."""

    durable = True

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        identity = self.root / "service.json"
        if identity.exists():
            self.service_id = json.loads(identity.read_text())["service_id"]
        else:
            self.service_id = uuid4().hex
            _atomic_write(identity, json.dumps({"service_id": self.service_id}).encode())

    def _json(self, path, record):
        _atomic_write(path, json.dumps(record, indent=1, sort_keys=True).encode())

    def save_operation(self, dataset, job):
        self._json(self.root / "operations" / dataset / f"{job['id']}.json", job)

    def operations(self, dataset):
        folder = self.root / "operations" / dataset
        return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))] if folder.exists() else []

    def save_record(self, kind, dataset, record):
        self._json(self.root / f"{kind}s" / f"{dataset}.json", record)

    def record(self, kind, dataset):
        path = self.root / f"{kind}s" / f"{dataset}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def save_result(self, dataset, version, manifest):
        self._json(self.root / "results" / dataset / f"{version}.json", manifest)

    def result(self, dataset, version):
        path = self.root / "results" / dataset / f"{version}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def save_model(self, dataset, model, record):
        data = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        digest = hashlib.sha256(data).hexdigest()
        _atomic_write(self.root / "models" / dataset / f"{digest}.pkl", data)
        self.save_record("model", dataset, dict(record, model_sha256=digest))
        for stale in (self.root / "models" / dataset).glob("*.pkl"):
            if stale.stem != digest:
                stale.unlink()

    def load_model(self, dataset):
        record = self.record("model", dataset)
        if record is None:
            return None, None
        path = self.root / "models" / dataset / f"{record['model_sha256']}.pkl"
        data = path.read_bytes() if path.exists() else b""
        if hashlib.sha256(data).hexdigest() != record["model_sha256"]:
            return None, dict(record, error="model file is missing or does not match its recorded digest")
        return pickle.loads(data), record


def open_store(state_dir):
    return DirectoryStore(state_dir) if state_dir else MemoryStore()
