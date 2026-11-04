# cellcanvas

Please find the comprehensive napari-based prototype of cellcanvas here: https://github.com/cellcanvas/cellcanvas-prototype

And please check back to find the full version of cellcanvas

For the Copick workflows, install `pip install '.[server,client]'` from this checkout.
Run `python workflows/cellcanvas-server/server.py CONFIG.json`, then
`python workflows/napari-client/client.py --config_path CONFIG.json`.
These legacy workflows use Zarr 2; the local napari-cellcanvas workflow uses Zarr 3.
Superpixel creation needs the optional `cellcanvas-spp` package.
