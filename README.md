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
`python -m cellcanvas_service.prepare TOMOGRAM FEATURES MANIFEST OUT --name NAME [--labels LABELS.json]` registers a copick OME-Zarr tomogram with saved features on its grid; the manifest must declare a public model.
`scripts/extract_dinov2_features.py` produces such features with public DINOv2 weights.

### Labels, seeds and results

A dataset may declare its classes: `"labels": {"classes": {"1": "background", "2": "membrane"}, "background": 1}`.
Painting value 0 always means unknown and is never trained on; painting with undeclared values is refused.
A painting upload may include the `scale` and `translation` (angstrom, z/y/x) of the grid it was made on; another grid is refused with `geometry_mismatch`.

With a declared background class and a cellcanvas-core that provides `SeededDataManager` (cellcanvas-core#20), training can add inferred background rows from unknown voxels: `POST /datasets/NAME/train` with `{"label_policy": {"policy": "inferred_background", "parameters": {"max_seeds": 2000}}}`, or set it as the dataset's default `label_policy`; `null` disables it.
Seeds are never written into the painting. The model record (`GET /datasets/NAME/model`) lists the policy, parameters and a digest of the seed indices, and `GET /datasets/NAME/inferred` recomputes them and refuses if they no longer reproduce.
`"classifier": "rf"` or `"fewshot"` picks the classifier for one training run; `GET /datasets/NAME` lists the classifiers, label policies and outputs this dataset supports.

Every completed prediction has a result record, `GET /datasets/NAME/results/N`: the digest of the stored voxels (also sent as `X-Result-Digest`), the painting revision and digest it was trained on, the classifier, inferred seeds, labels, grid and feature provenance without local paths.
`examples/copick_roundtrip.py` shows a copick client: upload a copick painting with its grid, train, predict, check the digest and write the prediction as a new copick segmentation.

### Restart, providers and deployment

`--state-dir DIR` keeps the service ID, operation receipts, trained models and result records across restarts.
Work that was running when the process stopped is reported as failed with code `interrupted`, and an unfinished prediction write is never served.
Feature providers other than saved public features are separate packages registering a `cellcanvas.providers` entry point; they load only when a deployment profile lists them, and private ones also need `"allow_private": true`.
Promptable segmentation (for example an nnInteractive server) is a separate backend; profiles list its URL under `prompt_backends` and the service does not proxy it.
`profiles/` holds local, Reef, Bruno and AWS deployment profiles; `python -m cellcanvas_service.profiles check PROFILE [CONFIG]` validates one and resolves dataset locations, and `python -m cellcanvas_service --profile PROFILE` runs with it.
The service has no authentication: bind beyond loopback only behind an authenticating proxy (`"exposure": "behind_proxy"`).
Object-store (`s3://`) image and feature locations need a cellcanvas-core whose `DataSet.from_paths` accepts `storage_options`.
