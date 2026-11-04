"""Serve registered core datasets without viewer or Copick dependencies.

Operations run in worker threads and serialize per dataset. A disconnected
request does not cancel core execution. Models live in memory; prediction
arrays are persisted by core, whose writes can be partial after a failure.
Operation receipts are retained in memory for this service instance only.
"""

from dataclasses import dataclass, field
from threading import Lock, Thread
from uuid import UUID, uuid4

from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager
from sklearn.ensemble import RandomForestClassifier
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.routing import Route


def default_manager(dataset):
    return SemanticSegmentationManager(
        DataManager([dataset]), RandomForestClassifier(random_state=0, n_jobs=1))


class ServiceError(Exception):
    def __init__(self, message, code, status):
        super().__init__(message)
        self.code, self.status = code, status


@dataclass
class DatasetState:
    dataset: object
    manager: object
    trained: bool = False
    prediction_status: str = "unknown"
    lock: object = field(default_factory=Lock)
    jobs: dict = field(default_factory=dict)
    jobs_lock: object = field(default_factory=Lock)

    def describe(self):
        dataset = self.dataset
        geometry = dataset.spatial_transform
        feature_axes = ["z", "y", "x"]
        feature_axes.insert(dataset.feature_axis % 4, "d")
        return {
            "image": {"axes": ["c", "z", "y", "x"], "shape": list(dataset.image.shape)},
            "painting": {"axes": ["z", "y", "x"], "shape": list(dataset.labels.shape)},
            "prediction": {"axes": ["z", "y", "x"], "shape": list(dataset.segmentation.shape),
                           "status": self.prediction_status},
            "features": {"axis": dataset.feature_axis, "axes": feature_axes,
                         "shapes": [list(values.shape) for values in dataset.features.values()]},
            "spatial_transform": {"scale": list(geometry.scale), "translation": list(geometry.translation),
                                  "units": list(geometry.units)},
            "trained": self.trained,
            "busy": self.lock.locked(),
            "storage": self.storage(),
        }

    def storage(self):
        from zarr.storage import LocalStore

        def location(array):
            if isinstance(getattr(array, "store", None), LocalStore):
                return {"uri": array.store.root.resolve().as_uri(), "path": array.path}
            return None

        return {"image": location(self.dataset.image), "painting": location(self.dataset.labels),
                "prediction": location(self.dataset.segmentation),
                "features": [location(array) for array in self.dataset.features.values()]}

    def operation(self, operation_id):
        with self.jobs_lock:
            if operation_id not in self.jobs:
                raise ServiceError("Operation not found.", "not_found", 404)
            return dict(self.jobs[operation_id])

    def submit(self, operation_id, operation, batch_size):
        with self.jobs_lock:
            if operation_id in self.jobs:
                previous = self.jobs[operation_id]
                if (previous["operation"], previous["batch_size"]) != (operation, batch_size):
                    raise ServiceError("Operation ID was already used with different options.", "operation_conflict", 409)
                return dict(previous)
            if not self.lock.acquire(blocking=False):
                raise ServiceError("A dataset operation is already running.", "busy", 409)
            job = {"id": operation_id, "operation": operation, "batch_size": batch_size,
                   "state": "running", "result": None, "error": None}
            self.jobs[operation_id] = job

        def run():
            try:
                result = self.execute(operation, batch_size, reserved=True)
                update = {"state": "succeeded", "result": result}
            except ServiceError as error:
                update = {"state": "failed", "error": {"message": str(error), "code": error.code}}
            with self.jobs_lock:
                job.update(update)

        try:
            # Non-daemon: normal process exit waits for accepted core work.
            Thread(target=run).start()
        except Exception as error:
            self.lock.release()
            with self.jobs_lock:
                job.update(state="failed", error={"message": str(error), "code": "core_error"})
            raise ServiceError(str(error), "core_error", 500) from error
        return self.operation(operation_id)

    def execute(self, operation, batch_size=65536, *, reserved=False):
        # Acquire and release inside the worker, so request cancellation cannot
        # release the model/storage lock while core is still running.
        if not reserved and not self.lock.acquire(blocking=False):
            raise ServiceError("A dataset operation is already running.", "busy", 409)
        try:
            if operation == "train":
                self.trained = False
                self.prediction_status = "unknown"
                self.manager.fit()
                self.trained = True
            else:
                if not self.trained:
                    raise ServiceError("Train this dataset before predicting.", "not_trained", 409)
                self.prediction_status = "writing"
                try:
                    self.manager.predict_to_segmentation(self.dataset, batch_size=batch_size)
                except Exception:
                    self.prediction_status = "failed"
                    raise
                self.prediction_status = "ready"
            return {"status": "success", "operation": operation, "trained": self.trained,
                    "prediction_status": self.prediction_status}
        except ServiceError:
            raise
        except Exception as error:
            # Expected core failures are handled responses, so clients can
            # inspect state or retry on the same HTTP connection.
            raise ServiceError(str(error), "core_error", 500) from error
        finally:
            self.lock.release()


async def options(request, allowed):
    try:
        body = await request.body()
        data = await request.json() if body else {}
    except ValueError:
        raise ServiceError("Request body must be a JSON object.", "invalid_request", 400) from None
    if not isinstance(data, dict) or set(data) - allowed:
        raise ServiceError("Request contains unsupported options.", "invalid_request", 400)
    return data


def create_app(datasets, *, manager_factory=default_manager):
    """Register server-owned DataSet objects; model construction is injectable.

    The service owns each manager and serializes its operations. Callers must
    serialize external access to the same model, painting or prediction stores.
    """
    if not datasets or any(not isinstance(name, str) or not name or "/" in name for name in datasets):
        raise ValueError("Register at least one dataset with nonempty names containing no slash.")
    states = {name: DatasetState(dataset, manager_factory(dataset)) for name, dataset in datasets.items()}
    service_id = uuid4().hex

    def metadata(state):
        return dict(state.describe(), service_id=service_id)

    def state_for(request):
        try:
            return states[request.path_params["name"]]
        except KeyError:
            raise ServiceError("Dataset not found.", "not_found", 404) from None

    async def health(request):
        return JSONResponse({"status": "ok"})

    async def list_datasets(request):
        return JSONResponse({"datasets": {name: metadata(state) for name, state in states.items()}})

    async def describe(request):
        return JSONResponse(metadata(state_for(request)))

    def operation_id(data):
        if "operation_id" not in data and "service_id" not in data:
            return None
        if data.get("service_id") != service_id:
            raise ServiceError("Service instance changed; reconnect before starting work.", "service_changed", 409)
        try:
            return UUID(data["operation_id"]).hex
        except (KeyError, TypeError, ValueError, AttributeError):
            raise ServiceError("operation_id must be a UUID string.", "invalid_request", 400) from None

    async def operation_status(request):
        state = state_for(request)
        return JSONResponse(dict(state.operation(request.path_params["operation_id"]), service_id=service_id))

    async def train(request):
        state = state_for(request)
        data = await options(request, {"operation_id", "service_id"})
        identifier = operation_id(data)
        if identifier is not None:
            return JSONResponse(dict(state.submit(identifier, "train", 65536), service_id=service_id), status_code=202)
        result = await run_in_threadpool(state.execute, "train")
        return JSONResponse(result)

    async def predict(request):
        state = state_for(request)
        data = await options(request, {"batch_size", "operation_id", "service_id"})
        batch_size = data.get("batch_size", 65536)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ServiceError("batch_size must be a positive integer.", "invalid_request", 400)
        identifier = operation_id(data)
        if identifier is not None:
            return JSONResponse(dict(state.submit(identifier, "predict", batch_size), service_id=service_id), status_code=202)
        result = await run_in_threadpool(state.execute, "predict", batch_size)
        return JSONResponse(result)

    async def service_error(request, error):
        return JSONResponse({"error": str(error), "code": error.code}, status_code=error.status)

    app = Starlette(routes=[
        Route("/health", health),
        Route("/datasets", list_datasets),
        Route("/datasets/{name}", describe),
        Route("/datasets/{name}/train", train, methods=["POST"]),
        Route("/datasets/{name}/predict", predict, methods=["POST"]),
        Route("/datasets/{name}/operations/{operation_id}", operation_status),
    ], exception_handlers={ServiceError: service_error})
    app.state.datasets = states
    return app
