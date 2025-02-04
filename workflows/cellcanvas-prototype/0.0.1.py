# /// script
# title = "cellcanvas"
# description = "Run CellCanvas, a tool for cellular image analysis."
# author = "Kyle Harrington <cellcanvas@kyleharrington.com>, Kevin Yamauchi"
# license = "MIT"
# version = "0.0.2"
# keywords = ["imaging", "cryoet", "napari", "cellcanvas", "visualization"]
# repository = "https://github.com/cellcanvas/cellcanvas"
# documentation = "https://github.com/cellcanvas/cellcanvas#readme"
# homepage = "https://cellcanvas.org"
# classifiers = [
#     "Development Status :: 3 - Alpha",
#     "Intended Audience :: Science/Research",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.10",
#     "Topic :: Scientific/Engineering :: Bio-Informatics",
#     "Topic :: Scientific/Engineering :: Image Processing"
# ]
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11",
#     "mpfr",
#     "gmp",
#     "cgal",
#     "numpy",
#     "scipy",
#     "scikit-image",
#     "scikit-learn",
#     "matplotlib",
#     "pandas",
#     "jupyter",
#     "notebook",
#     "jupytext",
#     "quantities",
#     "ipywidgets",
#     "vispy",
#     "meshio",
#     "zarr",
#     "xarray",
#     "PyQt5>=5.13.2,!=5.15.0",
#     "lxml_html_clean",
#     "pyopencl",
#     "reikna",
#     "jupyterlab",
#     "torch",
#     "einops",
#     "fire",
#     "pillow",
#     "imagecodecs",
#     "bokeh>=2.4.2,<3",
#     "napari-segment-blobs-and-things-with-membranes",
#     "s3fs",
#     "fsspec",
#     "pooch",
#     "qtpy",
#     "superqt",
#     "yappi",
#     "ftfy",
#     "tqdm",
#     "imageio",
#     "pyarrow",
#     "squidpy",
#     "h5py",
#     "tifffile",
#     "nilearn",
#     "opencv-python-headless>=0.4.8",
#     "ome-zarr>=0.8.0",
#     "pydantic-ome-ngff>=0.2.3",
#     "python-dotenv>=0.21",
#     "xgboost>=2",
#     "mrcfile",
#     "starfile>=0.5.0",
#     "imodmodel>=0.0.7",
#     "typer",
#     "cellcanvas @ git+https://github.com/cellcanvas/cellcanvas-prototype",
#     "napari @ git+https://github.com/napari/napari.git@bd82daf209e17e6192045bb290c125e7edde40cb"
# ]
# ///

import typer
import napari
import os
from typing import Optional, List
from cellcanvas import CellCanvasApp

app = typer.Typer(help="Launch CellCanvas with optional preloaded files and configuration.")

@app.command()
def launch(
    files: Optional[List[str]] = typer.Option(None, help="Paths to files to load into CellCanvas."),
    display_mode: str = typer.Option("2d", help="Set viewer mode: '2d' or '3d'."),
):
    """
    Launch CellCanvas with optional file loading capabilities.
    """
    viewer = napari.Viewer()

    typer.echo("Launching napari...")
    napari.run()

if __name__ == "__main__":
    app()