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
An optional `"model"` key picks the classifier: `"rf"` (random forest, the default) or `"fewshot"` (core's `FewShotClassifier`, when the installed core has it).
Use `GET /datasets`, `POST /datasets/NAME/train` and `POST /datasets/NAME/predict` with optional `{"batch_size": 65536}`.
