# /// script
# title = "CellCanvas-enabled Copick Server"
# description = "Run a Copick server with CellCanvas VAE feature extraction and reconstruction capabilities"
# author = "Kyle Harrington"
# license = "MIT"
# version = "0.1.0"
# keywords = ["server", "deep learning", "cell imaging", "microscopy"]
# repository = "https://github.com/kephale/cellcanvas"
# documentation = "https://github.com/kephale/cellcanvas#readme"
# classifiers = [
#     "Development Status :: 4 - Beta",
#     "Intended Audience :: Science/Research",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.9",
#     "Topic :: Scientific/Engineering :: Bio-Informatics",
#     "Topic :: Scientific/Engineering :: Visualization",
# ]
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
#     "torch",
#     "copick",
#     "copick-server @ git+https://github.com/kephale/copick-server",
#     "click",
#     "uvicorn",
#     "starlette",
#     "zarr<3",
# ]
# ///

import os
import json
import click
import uvicorn
import copick
import numpy as np
from pathlib import Path
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response, JSONResponse
from starlette.routing import Mount, Route
from starlette.requests import Request

# Import original Copick server components
from copick_server.server import CopickRoute, create_copick_app

class CellCanvasRoute:
    """Route handler for CellCanvas-specific functionality."""
    
    def __init__(self, root):
        self.root = root
    
    async def handle_painting_update(self, request):
        """Handle painting updates from the Napari client."""
        try:
            data = await request.json()
            
            # Extract data from request
            run_name = data.get('run_name')
            voxel_size = data.get('voxel_size')
            user_id = data.get('user_id', 'napariUser')
            session_id = data.get('session_id', '0')
            segmentation_name = data.get('segmentation_name', 'painting')
            coordinates = data.get('coordinates', [])
            label = data.get('label', 0)
            
            print(f"Received painting update for run {run_name}, coordinates: {len(coordinates)} points with label {label}")
            
            # Get the run
            run = self.root.get_run(run_name)
            if run is None:
                return JSONResponse({"error": f"Run {run_name} not found"}, status_code=404)
            
            # Get the segmentation
            segmentations = run.get_segmentations(
                voxel_size=voxel_size,
                name=segmentation_name,
                user_id=user_id,
                session_id=session_id,
                is_multilabel=True
            )
            
            if not segmentations:
                return JSONResponse({"error": "Segmentation not found"}, status_code=404)
            
            segmentation = segmentations[0]
            
            # Open the zarr array for modification
            zarr_file = segmentation.zarr()
            if "data" in zarr_file:
                data_arr = zarr_file["data"]
            else:
                data_arr = zarr_file["0"]
            
            # Update the segmentation with the new coordinates
            for coord in coordinates:
                try:
                    z, y, x = int(coord[0]), int(coord[1]), int(coord[2])
                    if 0 <= z < data_arr.shape[0] and 0 <= y < data_arr.shape[1] and 0 <= x < data_arr.shape[2]:
                        data_arr[z, y, x] = label
                except (IndexError, ValueError) as e:
                    print(f"Error updating coordinate {coord}: {str(e)}")
                    continue
            
            return JSONResponse({"status": "success"}, status_code=200)
            
        except Exception as e:
            print(f"Error handling painting update: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)
    
    async def ensure_segmentation(self, request):
        """Ensure a segmentation exists for a given tomogram, creating it if needed."""
        try:
            data = await request.json()
            
            # Extract data from request
            run_name = data.get('run_name')
            voxel_size = data.get('voxel_size')
            tomogram_type = data.get('tomogram_type')
            user_id = data.get('user_id', 'napariUser')
            session_id = data.get('session_id', '0')
            
            # Default segmentation names
            painting_name = data.get('painting_name', 'painting')
            prediction_name = data.get('prediction_name', 'prediction')
            
            print(f"Ensuring segmentations for run {run_name}, tomogram {tomogram_type}")
            
            # Get the run
            run = self.root.get_run(run_name)
            if run is None:
                return JSONResponse({"error": f"Run {run_name} not found"}, status_code=404)
            
            # Get the voxel spacing
            vs = run.get_voxel_spacing(voxel_size)
            if vs is None:
                return JSONResponse({"error": f"Voxel spacing {voxel_size} not found"}, status_code=404)
            
            # Get the tomogram
            tomogram = vs.get_tomogram(tomogram_type)
            if tomogram is None:
                return JSONResponse({"error": f"Tomogram {tomogram_type} not found"}, status_code=404)
            
            # Get the shape from the tomogram
            import zarr
            tomo_zarr = zarr.open(tomogram.zarr(), "r")
            if "0" in tomo_zarr:
                shape = tomo_zarr["0"].shape
            else:
                # Try to find any dataset to get shape
                for key in tomo_zarr.keys():
                    if isinstance(tomo_zarr[key], zarr.core.Array):
                        shape = tomo_zarr[key].shape
                        break
                else:
                    return JSONResponse({"error": "Could not determine tomogram shape"}, status_code=500)
            
            # Check/create painting segmentation
            painting_seg = None
            painting_segs = run.get_segmentations(
                voxel_size=voxel_size,
                name=painting_name,
                user_id=user_id,
                session_id=session_id,
                is_multilabel=True
            )
            
            if painting_segs:
                painting_seg = painting_segs[0]
                painting_created = False
            else:
                # Create new painting segmentation
                painting_seg = run.new_segmentation(
                    voxel_size=voxel_size,
                    name=painting_name,
                    session_id=session_id,
                    is_multilabel=True,
                    user_id=user_id
                )
                
                # Initialize empty segmentation
                zarr_file = zarr.open(painting_seg.zarr(), mode="w")
                zarr_file.create_dataset(
                    "data",
                    shape=shape,
                    dtype=np.int32,
                    chunks=(128, 128, 128),
                    fill_value=0
                )
                painting_created = True
            
            # Check/create prediction segmentation
            prediction_seg = None
            prediction_segs = run.get_segmentations(
                voxel_size=voxel_size,
                name=prediction_name,
                user_id=user_id,
                session_id=session_id,
                is_multilabel=True
            )
            
            if prediction_segs:
                prediction_seg = prediction_segs[0]
                prediction_created = False
            else:
                # Create new prediction segmentation
                prediction_seg = run.new_segmentation(
                    voxel_size=voxel_size,
                    name=prediction_name,
                    session_id=session_id,
                    is_multilabel=True,
                    user_id=user_id
                )
                
                # Initialize empty segmentation
                zarr_file = zarr.open(prediction_seg.zarr(), mode="w")
                zarr_file.create_dataset(
                    "data",
                    shape=shape,
                    dtype=np.int32,
                    chunks=(128, 128, 128),
                    fill_value=0
                )
                prediction_created = True
            
            return JSONResponse({
                "status": "success",
                "cellcanvas_tomogram": {
                    "run_name": run_name,
                    "voxel_size": voxel_size,
                    "tomogram_type": tomogram_type,
                    "user_id": user_id,
                    "session_id": session_id,
                    "painting_name": painting_name,
                    "prediction_name": prediction_name
                }
            }, status_code=200)
            
        except Exception as e:
            print(f"Error ensuring segmentations: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)
    
    async def ping(self, request):
        """Simple ping endpoint to check if server is running."""
        return JSONResponse({"status": "ok"}, status_code=200)

def create_cellcanvas_copick_app(root, cors_origins=None):
    """Create a Starlette app that includes both Copick and CellCanvas routes."""
    # Create the route handlers
    copick_handler = CopickRoute(root)
    cellcanvas_handler = CellCanvasRoute(root)
    
    routes = [
        # CellCanvas specific routes
        Route("/ping", endpoint=cellcanvas_handler.ping, methods=["GET"]),
        Route("/api/painting/update", endpoint=cellcanvas_handler.handle_painting_update, methods=["POST"]),
        Route("/api/segmentation/ensure", endpoint=cellcanvas_handler.ensure_segmentation, methods=["POST"]),
        
        # Debug route
        Route("/debug", endpoint=lambda request: Response(
            json.dumps({
                "runs": [run.meta.name for run in root.runs] if hasattr(root, "runs") else [],
                "root_type": str(type(root).__name__),
            }),
            media_type="application/json"
        )),

        # Original Copick catchall route - KEEP AT END
        Route("/{path:path}", endpoint=copick_handler.handle_request, methods=["GET", "HEAD", "PUT"]),
    ]
    
    # Create Starlette app
    app = Starlette(routes=routes)
    
    # Add CORS middleware if needed
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    return app

def serve_cellcanvas_copick(config_path, allowed_origins=None, **kwargs):
    """Start an HTTP server serving a Copick project with CellCanvas extensions."""
    root = copick.from_file(config_path)
    app = create_cellcanvas_copick_app(root, allowed_origins)
    uvicorn.run(app, **kwargs)

@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option(
    "--cors",
    type=str,
    default="*",  # Changed default to allow all origins
    help="Origin to allow CORS. Use wildcard '*' to allow all.",
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Bind socket to this host.",
    show_default=True,
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Bind socket to this port.",
    show_default=True,
)
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload.")
def main(config, cors, host, port, reload):
    """Serve a Copick project with CellCanvas features over HTTP."""
    print(f"Starting CellCanvas-enabled Copick server with config: {config}")

    serve_cellcanvas_copick(
        config,
        allowed_origins=[cors] if cors else ["*"],
        host=host,
        port=port,
        reload=reload
    )

if __name__ == "__main__":
    main()