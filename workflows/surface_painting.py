# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mrcfile",
#     "napari[all]",
#     "scikit-image",
#     "surforama",
#     "trimesh[easy]",
# ]
# ///

"""This workflow demonstrates how to pick particles on a surface by training a classifier."""

import mrcfile
import napari
from surforama.io import read_obj_file

if __name__ == "__main__":
    tomogram_file_path = "S1_M3b_StII_grow2_1_mesh_data.mrc"
    segmentation_file_path = "S1_M3b_StII_grow2_1_mesh_data_seg.mrc"
    mesh_file_path = "S1_M3b_StII_grow2_1_mesh_data.obj"

    # load the tomogram
    tomogram = mrcfile.read(tomogram_file_path)

    # load the segmentation
    segmentation = mrcfile.read(segmentation_file_path)

    # load the mesh
    mesh_data = read_obj_file(mesh_file_path)

    # Create a napari viewer
    viewer = napari.Viewer()

    # Add an data to the viewer
    viewer.add_image(tomogram, name="tomgoram")
    viewer.add_labels(segmentation, name="segmentation")
    viewer.add_surface(mesh_data)

    # Run the viewer
    napari.run()
