# cellcanvas

Please find the comprehensive napari-based prototype of cellcanvas here: https://github.com/cellcanvas/cellcanvas-prototype

And please check back to find the full version of cellcanvas

For the Copick workflows, install `pip install '.[server,client]'` from this checkout.
Run `python workflows/cellcanvas-server/server.py CONFIG.json`, then
`python workflows/napari-client/client.py --config_path CONFIG.json`.
These legacy workflows use Zarr 2; the local napari-cellcanvas workflow uses Zarr 3.
Superpixel creation needs the optional `cellcanvas-spp` package.

For the headless Zarr 3 service, install `pip install '.[service]'` in a separate environment.
Run `python -m cellcanvas_service CONFIG.json`; `"datasets"` maps names to core `DataSet.from_paths` arguments (paths are relative to the working directory).
Use `GET /datasets`, `POST /datasets/NAME/train` and `POST /datasets/NAME/predict` with optional `{"batch_size": 65536}`.
An external annotation store can replace a dataset's painting with `PUT /datasets/NAME/painting` (sparse flat indices and positive class IDs, a revision string and the current `service_id`), under the dataset lock; the model records the painting revision it was trained on.
`GET /datasets/NAME/prediction?version=N` returns a prediction as raw C-order bytes only when it completed and is current, so clients never publish a partial write.
`python -m cellcanvas_service.prepare TOMOGRAM FEATURES MANIFEST OUT --name NAME` registers a copick OME-Zarr tomogram with saved features on its grid; the manifest must declare a public model.
