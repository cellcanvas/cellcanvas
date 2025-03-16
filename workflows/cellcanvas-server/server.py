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
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch",
#     "copick",
#     "pandas",
#     "cellcanvas-spp @ file:///Users/kharrington/git/cellcanvas/superpixels",
#     "copick-server @ file:///Users/kharrington/git/copick/copick-server",
#     "copick-utils @ git+https://github.com/copick/copick-utils",
#     "click",
#     "uvicorn",
#     "starlette",
#     "zarr<3",
# ]
# ///

#     "cellcanvas-spp @ git+https://github.com/cellcanvas/superpixels",
#     "copick-server @ git+https://github.com/kephale/copick-server",

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

from cellcanvas_spp.segmentation import superpixels, superpixels_hws
import numpy as np
import zarr
from skimage.measure import regionprops_table
import pandas as pd

# Import original Copick server components
from copick_server.server import CopickRoute, create_copick_app

def import_time():
    """Helper function to import time module and return current time."""
    from datetime import datetime
    return datetime.now().isoformat()

class CellCanvasRoute:
    """Route handler for CellCanvas-specific functionality."""
    
    def __init__(self, root):
        self.root = root
    
    async def ping(self, request):
        """Simple ping endpoint to check if server is running."""
        try:
            # You can add any server status information here
            server_info = {
                "status": "ok",
                "server": "CellCanvas Copick Server",
                "timestamp": import_time(),
                "routes": [
                    "/ping",
                    "/api/painting/update",
                    "/api/segmentation/ensure",
                    "/debug",
                    "/{path:path}"
                ]
            }
            return JSONResponse(server_info, status_code=200)
        except Exception as e:
            print(f"Error in ping method: {str(e)}")
            return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

    async def handle_painting_update(self, request):
        """Handle painting updates from the Napari client with enhanced debugging."""
        print("\n==== RECEIVED PAINTING UPDATE REQUEST ====")
        try:
            # Get raw request body for debugging
            raw_body = await request.body()
            print(f"Raw request body size: {len(raw_body)} bytes")
            
            # Parse JSON data
            data = await request.json()
            print(f"Parsed JSON data keys: {list(data.keys())}")
            
            # Extract data from request
            run_name = data.get('run_name')
            voxel_size = data.get('voxel_size')
            user_id = data.get('user_id', 'napariUser')
            session_id = data.get('session_id', '0')
            segmentation_name = data.get('segmentation_name', 'painting')
            coordinates = data.get('coordinates', [])
            label = data.get('label', 0)
            
            # Debug print parameters
            print(f"Run name: {run_name}")
            print(f"Voxel size: {voxel_size}")
            print(f"User ID: {user_id}")
            print(f"Session ID: {session_id}")
            print(f"Segmentation name: {segmentation_name}")
            print(f"Number of coordinates: {len(coordinates)}")
            print(f"Label: {label}")
            
            # Print a few sample coordinates if available
            if coordinates and len(coordinates) > 0:
                print(f"Sample coordinates (first 5): {coordinates[:5]}")
            
            # Validate input
            if not run_name:
                print("Error: Missing run_name parameter")
                return JSONResponse({"error": "Missing run_name parameter"}, status_code=400)
            if not voxel_size:
                print("Error: Missing voxel_size parameter")
                return JSONResponse({"error": "Missing voxel_size parameter"}, status_code=400)
            if not coordinates:
                print("Error: No coordinates provided")
                return JSONResponse({"error": "No coordinates provided"}, status_code=400)
            
            print(f"Received painting update for run {run_name}, coordinates: {len(coordinates)} points with label {label}")
            
            # Get the run
            run = self.root.get_run(run_name)
            if run is None:
                print(f"Error: Run {run_name} not found")
                return JSONResponse({"error": f"Run {run_name} not found"}, status_code=404)
            
            print(f"Found run: {run_name}")
            
            # Get the segmentation
            segmentations = run.get_segmentations(
                voxel_size=voxel_size,
                name=segmentation_name,
                user_id=user_id,
                session_id=session_id,
                is_multilabel=True
            )
            
            if not segmentations:
                print(f"Segmentation not found. Attempting to create new segmentation.")
                # If segmentation doesn't exist, we'll create it
                try:
                    # Ensure voxel_spacing exists
                    vs = run.get_voxel_spacing(voxel_size)
                    if vs is None:
                        print(f"Error: Voxel spacing {voxel_size} not found")
                        return JSONResponse({"error": f"Voxel spacing {voxel_size} not found"}, status_code=404)
                    
                    print(f"Found voxel spacing: {voxel_size}")
                    
                    # Get a tomogram to determine shape
                    if not vs.tomograms:
                        print("Error: No tomograms found to determine shape")
                        return JSONResponse({"error": "No tomograms found to determine shape"}, status_code=404)
                    
                    print(f"Found tomograms: {[t.meta.tomo_type for t in vs.tomograms]}")
                    
                    # Get the first tomogram's shape
                    import zarr
                    tomo_zarr = zarr.open(vs.tomograms[0].zarr(), "r")
                    print(f"Opened tomogram zarr with keys: {list(tomo_zarr.keys())}")
                    
                    if "0" in tomo_zarr:
                        shape = tomo_zarr["0"].shape
                        print(f"Found shape from '0' dataset: {shape}")
                    else:
                        # Try to find any dataset to get shape
                        shape = None
                        for key in tomo_zarr.keys():
                            if isinstance(tomo_zarr[key], zarr.core.Array):
                                shape = tomo_zarr[key].shape
                                print(f"Found shape from '{key}' dataset: {shape}")
                                break
                        
                        if shape is None:
                            print("Error: Could not determine tomogram shape")
                            return JSONResponse({"error": "Could not determine tomogram shape"}, status_code=500)
                    
                    # Create new segmentation
                    print(f"Creating new segmentation: {segmentation_name}")
                    segmentation = run.new_segmentation(
                        voxel_size=voxel_size,
                        name=segmentation_name,
                        session_id=session_id,
                        is_multilabel=True,
                        user_id=user_id
                    )
                    
                    # Initialize with zeros
                    print(f"Initializing zarr dataset with shape {shape}")
                    zarr_file = zarr.open(segmentation.zarr(), mode="w")
                    zarr_file.create_dataset(
                        "data",
                        shape=shape,
                        dtype="i4",
                        chunks=(128, 128, 128),
                        fill_value=0
                    )
                    
                    print(f"Created new segmentation: {segmentation_name}")
                except Exception as e:
                    print(f"Error creating segmentation: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return JSONResponse({"error": f"Failed to create segmentation: {str(e)}"}, status_code=500)
                
                # Try to get the segmentation again
                segmentations = run.get_segmentations(
                    voxel_size=voxel_size,
                    name=segmentation_name,
                    user_id=user_id,
                    session_id=session_id,
                    is_multilabel=True
                )
                
                if not segmentations:
                    print("Error: Failed to create or retrieve segmentation")
                    return JSONResponse({"error": "Failed to create or retrieve segmentation"}, status_code=500)
            
            segmentation = segmentations[0]
            print(f"Using segmentation: {segmentation.meta.name}, path: {str(segmentation.zarr())}")
            
            # Open the zarr array for modification
            import zarr
            zarr_file = zarr.open(segmentation.zarr(), mode="r+")
            print(f"Opened zarr file with keys: {list(zarr_file.keys())}")
            
            # Check if "data" exists, otherwise try "0"
            if "data" in zarr_file:
                data_arr = zarr_file["data"]
                print(f"Using 'data' dataset with shape: {data_arr.shape}")
            elif "0" in zarr_file:
                data_arr = zarr_file["0"]
                print(f"Using '0' dataset with shape: {data_arr.shape}")
            else:
                # Look for any array in the zarr file
                found = False
                for key in zarr_file.keys():
                    if isinstance(zarr_file[key], zarr.core.Array):
                        data_arr = zarr_file[key]
                        print(f"Using '{key}' dataset with shape: {data_arr.shape}")
                        found = True
                        break
                
                if not found:
                    print("Error: No data array found in segmentation zarr")
                    return JSONResponse({"error": "No data array found in segmentation zarr"}, status_code=500)
            
            # Validate shape
            if len(data_arr.shape) != 3:
                print(f"Error: Unexpected data shape: {data_arr.shape}")
                return JSONResponse({"error": f"Unexpected data shape: {data_arr.shape}"}, status_code=500)
            
            # Update the segmentation with the new coordinates
            update_count = 0
            error_count = 0
            
            print(f"Updating {len(coordinates)} coordinates with label {label}")
            for i, coord in enumerate(coordinates):
                try:
                    # Ensure we have 3 coordinates
                    if len(coord) != 3:
                        print(f"Invalid coordinate at index {i}: {coord} (expected 3 values)")
                        error_count += 1
                        continue
                    
                    z, y, x = int(coord[0]), int(coord[1]), int(coord[2])
                    
                    # Validate coordinate bounds
                    if 0 <= z < data_arr.shape[0] and 0 <= y < data_arr.shape[1] and 0 <= x < data_arr.shape[2]:
                        # Set the voxel to the specified label
                        data_arr[z, y, x] = label
                        update_count += 1
                        # Print periodic updates
                        if update_count % 100 == 0:
                            print(f"Updated {update_count} voxels so far...")
                    else:
                        error_count += 1
                        if error_count < 5:  # Limit the number of error messages
                            print(f"Coordinate out of bounds: [{z}, {y}, {x}], shape: {data_arr.shape}")
                except (IndexError, ValueError) as e:
                    print(f"Error updating coordinate {coord}: {str(e)}")
                    error_count += 1
                    continue
            
            print(f"Painting update completed: {update_count} voxels updated, {error_count} errors")
            
            result = {
                "status": "success",
                "updates": update_count,
                "errors": error_count,
                "label": label
            }
            
            if error_count > 0:
                print(f"Warning: {error_count} coordinates were outside valid range")
            
            print(f"Successfully updated {update_count} voxels with label {label}")
            return JSONResponse(result, status_code=200)
            
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
            
            # Validate input
            if not run_name:
                return JSONResponse({"error": "Missing run_name parameter"}, status_code=400)
            if not voxel_size:
                return JSONResponse({"error": "Missing voxel_size parameter"}, status_code=400)
            if not tomogram_type:
                return JSONResponse({"error": "Missing tomogram_type parameter"}, status_code=400)
            
            print(f"Ensuring segmentations for run {run_name}, tomogram {tomogram_type}, voxel size {voxel_size}")
            
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
            
            # Try to determine shape from the tomogram
            shape = None
            if "0" in tomo_zarr:
                shape = tomo_zarr["0"].shape
            else:
                # Try to find any dataset to get shape
                for key in tomo_zarr.keys():
                    if isinstance(tomo_zarr[key], zarr.core.Array):
                        shape = tomo_zarr[key].shape
                        break
            
            if shape is None:
                return JSONResponse({"error": "Could not determine tomogram shape"}, status_code=500)
            
            print(f"Tomogram shape: {shape}")
            
            # Function to ensure a segmentation exists
            def ensure_segmentation(name, is_multilabel=True):
                # Check if segmentation exists
                segs = run.get_segmentations(
                    voxel_size=voxel_size,
                    name=name,
                    user_id=user_id,
                    session_id=session_id,
                    is_multilabel=is_multilabel
                )
                
                if segs:
                    seg = segs[0]
                    # Verify the segmentation has data
                    seg_zarr = zarr.open(seg.zarr(), "r")
                    data_found = False
                    
                    # Look for either "data" or "0" datasets
                    if "data" in seg_zarr:
                        data_found = True
                    elif "0" in seg_zarr:
                        data_found = True
                    else:
                        # Check all keys for arrays
                        for key in seg_zarr.keys():
                            if isinstance(seg_zarr[key], zarr.core.Array):
                                data_found = True
                                break
                    
                    if data_found:
                        return seg, False  # Existing segmentation
                
                # Create new segmentation
                seg = run.new_segmentation(
                    voxel_size=voxel_size,
                    name=name,
                    session_id=session_id,
                    is_multilabel=is_multilabel,
                    user_id=user_id
                )
                
                # Initialize with zeros
                zarr_file = zarr.open(seg.zarr(), mode="w")
                zarr_file.create_dataset(
                    "data",  # Use "data" as the standard dataset name
                    shape=shape,
                    dtype="i4",  # Use int32 for labels
                    chunks=(128, 128, 128),
                    fill_value=0
                )
                
                return seg, True  # New segmentation
            
            # Ensure both segmentations exist
            painting_seg, painting_created = ensure_segmentation(painting_name)
            prediction_seg, prediction_created = ensure_segmentation(prediction_name)
            
            # Get string paths for zarr files instead of FSStore objects
            # Convert FSStore objects to strings to make them JSON serializable
            painting_path = str(painting_seg.zarr()) if painting_seg else None
            prediction_path = str(prediction_seg.zarr()) if prediction_seg else None
            
            # Prepare response with status information
            segmentation_info = {
                "painting": {
                    "name": painting_name,
                    "created": painting_created,
                    "path": painting_path
                },
                "prediction": {
                    "name": prediction_name,
                    "created": prediction_created,
                    "path": prediction_path
                }
            }
            
            # Convert shape to a list to make it JSON serializable
            shape_list = list(shape) if shape else None
            
            return JSONResponse({
                "status": "success",
                "message": f"Segmentations {'created' if painting_created or prediction_created else 'verified'}",
                "segmentations": segmentation_info,
                "cellcanvas_tomogram": {
                    "run_name": run_name,
                    "voxel_size": voxel_size,
                    "tomogram_type": tomogram_type,
                    "user_id": user_id,
                    "session_id": session_id,
                    "shape": shape_list
                }
            }, status_code=200)
                
        except Exception as e:
            print(f"Error ensuring segmentations: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)

    async def create_superpixel_segmentation(self, request):
        """Create a superpixel segmentation for a tomogram in the Copick project."""
        print("\n==== RECEIVED CREATE SUPERPIXEL SEGMENTATION REQUEST ====")
        try:
            # Parse JSON data
            data = await request.json()
            print(f"Parsed JSON data keys: {list(data.keys())}")
            
            # Extract data from request
            run_name = data.get('run_name')
            voxel_size = data.get('voxel_size')
            tomogram_type = data.get('tomogram_type', 'wbp')  # Default to wbp
            user_id = data.get('user_id', 'cellcanvasSPP')
            session_id = data.get('session_id', '0')
            segmentation_name = data.get('segmentation_name', 'superpixelSegmentation')
            
            # Superpixel parameters
            sigma = data.get('sigma', 0.25)
            h_minima = data.get('h_minima', None)
            use_hws = data.get('use_hws', True)  # Default to hierarchical watershed
            
            # Optional crop parameters
            crop_z = data.get('crop_z')
            crop_y = data.get('crop_y')
            crop_x = data.get('crop_x')
            
            # Validate input
            if not run_name:
                return JSONResponse({"error": "Missing run_name parameter"}, status_code=400)
            if not voxel_size:
                return JSONResponse({"error": "Missing voxel_size parameter"}, status_code=400)
            
            print(f"Creating superpixel segmentation for run {run_name}, voxel size {voxel_size}")
            
            # Get the run
            run = self.root.get_run(run_name)
            if run is None:
                return JSONResponse({"error": f"Run {run_name} not found"}, status_code=404)
            
            # Get the voxel spacing
            vs = run.get_voxel_spacing(voxel_size)
            if vs is None:
                return JSONResponse({"error": f"Voxel spacing {voxel_size} not found"}, status_code=404)
            
            # Get the tomogram
            tomogram = None
            for tomo in vs.tomograms:
                if tomo.meta.tomo_type == tomogram_type:
                    tomogram = tomo
                    break
            
            if tomogram is None:
                return JSONResponse({"error": f"Tomogram {tomogram_type} not found"}, status_code=404)
            
            # Check if segmentation already exists
            existing_segs = run.get_segmentations(
                voxel_size=voxel_size,
                name=segmentation_name,
                user_id=user_id,
                session_id=session_id,
                is_multilabel=True
            )
            
            if existing_segs:
                return JSONResponse({
                    "status": "exists",
                    "message": f"Segmentation '{segmentation_name}' already exists",
                    "segmentation_path": str(existing_segs[0].zarr())
                }, status_code=200)
            
            # Open tomogram data
            tomo_zarr = zarr.open(tomogram.zarr(), "r")
            # Find data array
            data_array = None
            for key in tomo_zarr.keys():
                if isinstance(tomo_zarr[key], zarr.core.Array):
                    data_array = tomo_zarr[key]
                    break
            
            if data_array is None:
                return JSONResponse({"error": "Could not find data array in tomogram"}, status_code=500)
            
            print(f"Loading tomogram data with shape {data_array.shape}")
            
            # Apply crop if specified
            if crop_z or crop_y or crop_x:
                z_slice = slice(*crop_z) if crop_z else slice(None)
                y_slice = slice(*crop_y) if crop_y else slice(None)
                x_slice = slice(*crop_x) if crop_x else slice(None)
                
                img = data_array[z_slice, y_slice, x_slice]
                print(f"Cropped tomogram to shape {img.shape}")
            else:
                # Load the entire image (this could be memory-intensive for large tomograms)
                img = data_array[:]
            
            # Perform superpixel segmentation
            print(f"Performing superpixel segmentation with {'HWS' if use_hws else 'standard'} method, sigma={sigma}")
            
            if use_hws:
                segm = superpixels_hws(img, sigma=sigma)
            else:
                if h_minima is None:
                    h_minima = 0.0025  # Default value
                segm = superpixels(img, sigma=sigma, h_minima=h_minima)
            
            print(f"Superpixel segmentation complete with {np.max(segm)} regions")
            
            # Create new segmentation
            new_seg = run.new_segmentation(
                voxel_size=voxel_size,
                name=segmentation_name,
                session_id=session_id,
                is_multilabel=True,
                user_id=user_id
            )
            
            # Save segmentation to zarr
            segmentation_zarr = zarr.open(new_seg.zarr(), mode="w")
            segmentation_zarr.create_dataset(
                "data",
                data=segm,
                chunks=(128, 128, 128)
            )
            
            print(f"Segmentation saved to {new_seg.zarr()}")
            
            # Return success response
            return JSONResponse({
                "status": "success",
                "message": "Superpixel segmentation created successfully",
                "segmentation_info": {
                    "name": segmentation_name,
                    "path": str(new_seg.zarr()),
                    "region_count": int(np.max(segm)),
                    "shape": list(segm.shape)
                }
            }, status_code=200)
            
        except Exception as e:
            print(f"Error creating superpixel segmentation: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)

    async def compute_superpixel_dataframe(self, request):
        """Compute region properties for superpixel segmentation and return as dataframe."""
        print("\n==== RECEIVED COMPUTE SUPERPIXEL DATAFRAME REQUEST ====")
        try:
            # Parse JSON data
            data = await request.json()
            print(f"Parsed JSON data keys: {list(data.keys())}")
            
            # Extract data from request
            run_name = data.get('run_name')
            voxel_size = data.get('voxel_size')
            tomogram_type = data.get('tomogram_type', 'wbp')
            segmentation_name = data.get('segmentation_name', 'superpixelSegmentation')
            user_id = data.get('user_id', 'cellcanvasSPP')
            session_id = data.get('session_id', '0')
            
            # Optional crop parameters
            crop_z = data.get('crop_z')
            crop_y = data.get('crop_y')
            crop_x = data.get('crop_x')
            
            # Properties to compute
            properties = data.get('properties', [
                'label', 'area', 'bbox', 'bbox_area', 'centroid', 'equivalent_diameter', 'euler_number',
                'extent', 'filled_area', 'major_axis_length', 'max_intensity', 'mean_intensity', 
                'min_intensity'
            ])
            
            # Validate input
            if not run_name:
                return JSONResponse({"error": "Missing run_name parameter"}, status_code=400)
            if not voxel_size:
                return JSONResponse({"error": "Missing voxel_size parameter"}, status_code=400)
            
            print(f"Computing dataframe for superpixels in run {run_name}, voxel size {voxel_size}")
            
            # Get the run
            run = self.root.get_run(run_name)
            if run is None:
                return JSONResponse({"error": f"Run {run_name} not found"}, status_code=404)
            
            # Get the segmentation
            segs = run.get_segmentations(
                voxel_size=voxel_size,
                name=segmentation_name,
                user_id=user_id,
                session_id=session_id,
                is_multilabel=True
            )
            
            if not segs:
                return JSONResponse({
                    "error": f"Segmentation {segmentation_name} not found",
                    "suggestion": "Create superpixel segmentation first using /api/superpixels/create"
                }, status_code=404)
            
            # Get the tomogram for intensity measurements
            vs = run.get_voxel_spacing(voxel_size)
            if vs is None:
                return JSONResponse({"error": f"Voxel spacing {voxel_size} not found"}, status_code=404)
            
            tomogram = None
            for tomo in vs.tomograms:
                if tomo.meta.tomo_type == tomogram_type:
                    tomogram = tomo
                    break
            
            if tomogram is None:
                return JSONResponse({"error": f"Tomogram {tomogram_type} not found"}, status_code=404)
            
            # Open tomogram and segmentation data
            segmentation_zarr = zarr.open(segs[0].zarr(), "r")
            tomo_zarr = zarr.open(tomogram.zarr(), "r")
            
            # Find data arrays
            seg_array = None
            for key in segmentation_zarr.keys():
                if isinstance(segmentation_zarr[key], zarr.core.Array):
                    seg_array = segmentation_zarr[key]
                    break
                    
            img_array = None
            for key in tomo_zarr.keys():
                if isinstance(tomo_zarr[key], zarr.core.Array):
                    img_array = tomo_zarr[key]
                    break
            
            if seg_array is None:
                return JSONResponse({"error": "Could not find data array in segmentation"}, status_code=500)
            if img_array is None:
                return JSONResponse({"error": "Could not find data array in tomogram"}, status_code=500)
            
            # Apply crop if specified
            if crop_z or crop_y or crop_x:
                z_slice = slice(*crop_z) if crop_z else slice(None)
                y_slice = slice(*crop_y) if crop_y else slice(None)
                x_slice = slice(*crop_x) if crop_x else slice(None)
                
                seg = seg_array[z_slice, y_slice, x_slice]
                img = img_array[z_slice, y_slice, x_slice]
                print(f"Cropped data to shape {seg.shape}")
            else:
                # Load data in chunks if it's large
                if seg_array.size > 100_000_000:  # ~100 million elements
                    print("Segmentation is large, processing a subset for dataframe computation")
                    # Get a representative subset
                    mid_z = seg_array.shape[0] // 2
                    subset_z = slice(max(0, mid_z - 50), min(seg_array.shape[0], mid_z + 50))
                    seg = seg_array[subset_z, :, :]
                    img = img_array[subset_z, :, :]
                else:
                    seg = seg_array[:]
                    img = img_array[:]
            
            # Compute region properties
            print(f"Computing region properties with properties: {properties}")
            props = regionprops_table(seg, intensity_image=img, properties=properties)
            props_df = pd.DataFrame(props)
            
            # Add a painted_label column initialized to 0
            props_df['painted_label'] = 0
            
            # Convert centroid and bbox to serializable format if present
            if 'centroid' in props_df.columns:
                centroid_cols = [col for col in props_df.columns if col.startswith('centroid-')]
                props_df['centroid'] = props_df[centroid_cols].values.tolist()
                props_df = props_df.drop(columns=centroid_cols)
            
            if 'bbox' in props_df.columns:
                bbox_cols = [col for col in props_df.columns if col.startswith('bbox-')]
                props_df['bbox'] = props_df[bbox_cols].values.tolist()
                props_df = props_df.drop(columns=bbox_cols)
            
            # Convert dataframe to serializable dictionary
            result_dict = props_df.to_dict(orient='records')
            
            print(f"Region properties computed for {len(result_dict)} regions")
            
            # Return success response
            return JSONResponse({
                "status": "success",
                "message": "Superpixel dataframe computed successfully",
                "properties": list(props_df.columns),
                "dataframe": result_dict,
                "record_count": len(result_dict),
                "segmentation_info": {
                    "name": segmentation_name,
                    "path": str(segs[0].zarr())
                }
            }, status_code=200)
            
        except Exception as e:
            print(f"Error computing superpixel dataframe: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)

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
        
        # New superpixel endpoints
        Route("/api/superpixels/create", endpoint=cellcanvas_handler.create_superpixel_segmentation, methods=["POST"]),
        Route("/api/superpixels/dataframe", endpoint=cellcanvas_handler.compute_superpixel_dataframe, methods=["POST"]),
        
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

def serve_cellcanvas_copick(config_path=None, allowed_origins=None, **kwargs):
    """Start an HTTP server serving a Copick project with CellCanvas extensions."""
    if config_path:
        root = copick.from_file(config_path)
    else:
        root = copick.from_czcdp_datasets([10440], overlay_root="/tmp/test/")
    app = create_cellcanvas_copick_app(root, allowed_origins)
    uvicorn.run(app, **kwargs)

@click.command()
@click.argument("config", type=click.Path(exists=True), required=False)
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
def main(config=None, cors="*", host="127.0.0.1", port=8000, reload=False):
    """Serve a Copick project with CellCanvas features over HTTP."""
    if config:
        print(f"Starting CellCanvas-enabled Copick server with config: {config}")
        root = copick.from_file(config)
    else:
        print("No config provided, using default project with dataset 10440")
        root = copick.from_czcdp_datasets([10440], overlay_root="/tmp/test/")
    
    app = create_cellcanvas_copick_app(root, [cors] if cors else ["*"])
    uvicorn.run(app, host=host, port=port, reload=reload)

if __name__ == "__main__":
    main()