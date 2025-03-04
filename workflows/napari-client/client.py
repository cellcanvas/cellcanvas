# /// script
# title = "napari-copick-cellcanvas"
# description = "A Python script to launch napari with the napari-copick plugin integrated with CellCanvas server."
# author = "Kyle Harrington <napari@kyleharrington.com>"
# license = "MIT"
# version = "0.1.0"
# keywords = ["napari", "copick", "cellcanvas", "visualization", "plugin", "tomography", "cryo-EM"]
# repository = "https://github.com/kephale/napari-copick"
# documentation = "https://github.com/kephale/napari-copick#readme"
# homepage = "https://napari.org"
# classifiers = [
#     "Development Status :: 3 - Alpha",
#     "Intended Audience :: Developers",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.12",
#     "Topic :: Scientific/Engineering :: Visualization",
#     "Topic :: Scientific/Engineering :: Bio-Informatics",
# ]
# requires-python = ">=3.12"
# dependencies = [
#     "napari[all]",
#     "zarr<3",
#     "napari-copick @ git+https://github.com/kephale/napari-copick.git",
#     "requests",
#     "numpy",
#     "matplotlib",
# ]
# ///

import dask.array as da
import numpy as np
import copick
import zarr
from zarr.storage import LRUStoreCache
import napari
import sys
import requests
import json
from pathlib import Path
from qtpy.QtWidgets import (
    QWidget,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QLineEdit,
    QMenu,
    QAction,
    QFormLayout,
    QComboBox,
    QSpinBox,
    QGroupBox,
    QHBoxLayout,
    QCheckBox,
    QGridLayout,
    QDialog,
)
from qtpy.QtCore import Qt, QPoint
from napari.utils import DirectLabelColormap


class CellCanvasServerDialog(QDialog):
    """Dialog for CellCanvas server settings."""
    
    def __init__(self, parent=None, run_name=None, voxel_spacing=None):
        super().__init__(parent)
        self.setWindowTitle("CellCanvas Server Settings")
        self.setMinimumWidth(400)
        
        # Initialize with defaults or provided values
        self.server_url = "http://localhost:8000"
        self.run_name = run_name or ""
        self.voxel_spacing = voxel_spacing or 10.0
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Server settings
        server_group = QGroupBox("Server Settings")
        server_layout = QFormLayout()
        
        self.server_url_input = QLineEdit(self.server_url)
        server_layout.addRow("Server URL:", self.server_url_input)
        
        server_group.setLayout(server_layout)
        layout.addWidget(server_group)
        
        # Project settings
        proj_group = QGroupBox("Project Settings")
        proj_layout = QFormLayout()
        
        self.run_name_input = QLineEdit(self.run_name)
        proj_layout.addRow("Run Name:", self.run_name_input)
        
        self.voxel_spacing_input = QLineEdit(str(self.voxel_spacing))
        proj_layout.addRow("Voxel Spacing:", self.voxel_spacing_input)
        
        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def get_settings(self):
        """Return the current settings from the dialog."""
        return {
            'server_url': self.server_url_input.text(),
            'run_name': self.run_name_input.text(),
            'voxel_spacing': float(self.voxel_spacing_input.text())
        }


class CellCanvasClient:
    """Client for CellCanvas server API."""
    
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url
    
    def send_request(self, endpoint, params=None, method="GET"):
        """Send a request to the server."""
        url = f"{self.server_url}/{endpoint}"
        print(f"Sending {method} request to {url}")
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params)
            elif method.upper() == "POST":
                response = requests.post(url, json=params)
            else:
                print(f"Unsupported method: {method}")
                return None
                
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                return None
                
            return response.json()
        except Exception as e:
            print(f"Request error: {str(e)}")
            return None


class CopickPlugin(QWidget):
    def __init__(self, viewer=None, config_path=None):
        super().__init__()
        if viewer:
            self.viewer = viewer
        else:
            self.viewer = napari.Viewer()

        self.root = None
        self.selected_run = None
        self.current_layer = None
        self.session_id = "17"
        self.cellcanvas_client = CellCanvasClient()
        self.cellcanvas_settings = {
            'server_url': "http://localhost:8000",
            'run_name': "",
            'voxel_spacing': 10.0
        }
        
        # For tracking active tomogram
        self.active_tomogram = None
        self.active_tomogram_scale = None
        
        self.setup_ui()
        if config_path:
            self.load_config(config_path)

    def setup_ui(self):
        layout = QVBoxLayout()

        # Config loading button
        self.load_button = QPushButton("Load Config")
        self.load_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.load_button)

        # Hierarchical tree view
        self.tree_view = QTreeWidget()
        self.tree_view.setHeaderLabel("Copick Project")
        self.tree_view.itemExpanded.connect(self.handle_item_expand)
        self.tree_view.itemClicked.connect(self.handle_item_click)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(
            self.open_context_menu
        )
        layout.addWidget(self.tree_view)

        # CellCanvas integration section
        cellcanvas_group = QGroupBox("CellCanvas Integration")
        cellcanvas_layout = QVBoxLayout()
        
        self.cellcanvas_status_label = QLabel("CellCanvas: Not connected")
        cellcanvas_layout.addWidget(self.cellcanvas_status_label)
        
        # CellCanvas configuration button
        self.configure_cellcanvas_button = QPushButton("Configure CellCanvas")
        self.configure_cellcanvas_button.clicked.connect(self.configure_cellcanvas)
        cellcanvas_layout.addWidget(self.configure_cellcanvas_button)
        
        cellcanvas_group.setLayout(cellcanvas_layout)
        layout.addWidget(cellcanvas_group)

        # Info label
        self.info_label = QLabel("Select a pick to get started")
        layout.addWidget(self.info_label)

        self.setLayout(layout)

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Config", "", "JSON Files (*.json)"
        )
        if path:
            self.load_config(path)

    def load_config(self, path=None):
        if path:
            self.root = copick.from_file(path)
            self.populate_tree()
            
            # Update CellCanvas run name if available
            if self.root and self.root.runs:
                self.cellcanvas_settings['run_name'] = self.root.runs[0].meta.name

    def populate_tree(self):
        self.tree_view.clear()
        for run in self.root.runs:
            run_item = QTreeWidgetItem(self.tree_view, [run.meta.name])
            run_item.setData(0, Qt.UserRole, run)
            run_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)

    def handle_item_expand(self, item):
        data = item.data(0, Qt.UserRole)
        if isinstance(data, copick.models.CopickRun):
            self.expand_run(item, data)
        elif isinstance(data, copick.models.CopickVoxelSpacing):
            self.expand_voxel_spacing(item, data)

    def expand_run(self, item, run):
        if not item.childCount():
            for voxel_spacing in run.voxel_spacings:
                spacing_item = QTreeWidgetItem(
                    item, [f"Voxel Spacing: {voxel_spacing.meta.voxel_size}"]
                )
                spacing_item.setData(0, Qt.UserRole, voxel_spacing)
                spacing_item.setChildIndicatorPolicy(
                    QTreeWidgetItem.ShowIndicator
                )

            # Add picks nested by user_id, session_id, and pickable_object_name
            picks = run.picks
            picks_item = QTreeWidgetItem(item, ["Picks"])
            user_dict = {}
            for pick in picks:
                if pick.meta.user_id not in user_dict:
                    user_dict[pick.meta.user_id] = {}
                if pick.meta.session_id not in user_dict[pick.meta.user_id]:
                    user_dict[pick.meta.user_id][pick.meta.session_id] = []
                user_dict[pick.meta.user_id][pick.meta.session_id].append(pick)

            for user_id, sessions in user_dict.items():
                user_item = QTreeWidgetItem(picks_item, [f"User: {user_id}"])
                for session_id, picks in sessions.items():
                    session_item = QTreeWidgetItem(
                        user_item, [f"Session: {session_id}"]
                    )
                    for pick in picks:
                        pick_child = QTreeWidgetItem(
                            session_item, [pick.meta.pickable_object_name]
                        )
                        pick_child.setData(0, Qt.UserRole, pick)
            item.addChild(picks_item)

    def expand_voxel_spacing(self, item, voxel_spacing):
        if not item.childCount():
            tomogram_item = QTreeWidgetItem(item, ["Tomograms"])
            for tomogram in voxel_spacing.tomograms:
                tomo_child = QTreeWidgetItem(
                    tomogram_item, [tomogram.meta.tomo_type]
                )
                tomo_child.setData(0, Qt.UserRole, tomogram)
            item.addChild(tomogram_item)

            segmentation_item = QTreeWidgetItem(item, ["Segmentations"])
            segmentations = voxel_spacing.run.get_segmentations(
                voxel_size=voxel_spacing.meta.voxel_size
            )
            for segmentation in segmentations:
                seg_child = QTreeWidgetItem(
                    segmentation_item, [segmentation.meta.name]
                )
                seg_child.setData(0, Qt.UserRole, segmentation)
            item.addChild(segmentation_item)

    def handle_item_click(self, item, column):
        data = item.data(0, Qt.UserRole)
        if isinstance(data, copick.models.CopickRun):
            self.info_label.setText(f"Run: {data.meta.name}")
            self.selected_run = data
            # Update CellCanvas settings with this run
            self.cellcanvas_settings['run_name'] = data.meta.name
        elif isinstance(data, copick.models.CopickVoxelSpacing):
            self.info_label.setText(f"Voxel Spacing: {data.meta.voxel_size}")
            self.lazy_load_voxel_spacing(item, data)
            # Update CellCanvas settings with this voxel spacing
            self.cellcanvas_settings['voxel_spacing'] = data.meta.voxel_size
        elif isinstance(data, copick.models.CopickTomogram):
            self.load_tomogram(data)
            # Store the active tomogram for CellCanvas integration
            self.active_tomogram = data
        elif isinstance(data, copick.models.CopickSegmentation):
            self.load_segmentation(data)
        elif isinstance(data, copick.models.CopickPicks):
            parent_run = self.get_parent_run(item)
            self.load_picks(data, parent_run)

    def get_parent_run(self, item):
        while item:
            data = item.data(0, Qt.UserRole)
            if isinstance(data, copick.models.CopickRun):
                return data
            item = item.parent()
        return None

    def lazy_load_voxel_spacing(self, item, voxel_spacing):
        if not item.childCount():
            self.expand_voxel_spacing(item, voxel_spacing)

    def load_tomogram(self, tomogram):
        zarr_store = LRUStoreCache(tomogram.zarr(), max_size = 10_000_000_000)
        zarr_group = zarr.open(zarr_store, "r")

        # Determine the number of scale levels
        scale_levels = [key for key in zarr_group.keys() if key.isdigit()]
        scale_levels.sort(key=int)

        # Open the tomogram with napari-ome-zarr
        layer = self.viewer.add_image([zarr_group[level] for level in scale_levels])
        
        # Get the voxel size from the parent voxel spacing object
        # We need to find the parent voxel spacing that this tomogram belongs to
        voxel_size = None
        for run in self.root.runs:
            for voxel_spacing in run.voxel_spacings:
                if tomogram in voxel_spacing.tomograms:
                    voxel_size = voxel_spacing.meta.voxel_size
                    break
            if voxel_size is not None:
                break
        
        # Set scales for coordinate transformation
        if voxel_size is not None:
            self.active_tomogram_scale = [voxel_size] * 3
        else:
            # Default scale if we can't find the voxel size
            self.active_tomogram_scale = [1.0] * 3
            print("Warning: Could not find voxel size for tomogram, using default scale of 1.0")

        layer.scale = self.active_tomogram_scale

        self.info_label.setText(
            f"Loaded Tomogram: {tomogram.meta.tomo_type} with num scales = {len(scale_levels)}"
        )

        return layer

    def load_segmentation(self, segmentation):
        zarr_data = zarr.open(segmentation.zarr(), "r+")
        if "data" in zarr_data:
            data = zarr_data["data"]
        else:
            data = zarr_data["0"]

        scale = [segmentation.meta.voxel_size] * 3

        # Create a color map based on copick colors
        colormap = self.get_copick_colormap()
        
        # Create the labels layer with the correct colormap
        painting_layer = self.viewer.add_labels(
            data,
            name=f"Segmentation: {segmentation.meta.name}",
            scale=scale,
            opacity=0.5  # Add some transparency to make it easier to see
        )
        
        # Apply the colormap
        painting_layer.colormap = DirectLabelColormap(color_dict=colormap)
        
        # Set up the painting labels
        painting_layer.painting_labels = [
            obj.label for obj in self.root.config.pickable_objects
        ]
        
        # Store the class labels mapping
        self.class_labels_mapping = {
            obj.label: obj.name for obj in self.root.config.pickable_objects
        }

        self.info_label.setText(f"Loaded Segmentation: {segmentation.meta.name}")

    def get_copick_colormap(self, pickable_objects=None):
        if not pickable_objects:
            pickable_objects = self.root.config.pickable_objects
        
        colormap = {}
        for obj in pickable_objects:
            # Convert RGBA color values correctly
            color = np.array(obj.color)
            # Normalize RGB components to [0,1] range but keep alpha as is
            normalized_color = np.array([
                color[0] / 255.0,  # R
                color[1] / 255.0,  # G
                color[2] / 255.0,  # B
                color[3] / 255.0   # A
            ])
            colormap[obj.label-1] = normalized_color
        
        # Add background color
        colormap[None] = np.array([0, 0, 0, 0])  # transparent background
        return colormap

    def load_picks(self, pick_set, parent_run):
        if parent_run is not None:
            if pick_set:
                if pick_set.points:
                    points = [
                        (p.location.z, p.location.y, p.location.x)
                        for p in pick_set.points
                    ]
                    color = (
                        pick_set.color
                        if pick_set.color
                        else (255, 255, 255, 255)
                    )  # Default to white if color is not set
                    colors = np.tile(
                        np.array(
                            [
                                color[0] / 255.0,
                                color[1] / 255.0,
                                color[2] / 255.0,
                                color[3] / 255.0,
                            ]
                        ),
                        (len(points), 1),
                    )  # Create an array with the correct shape
                    pickable_object = [
                        obj
                        for obj in self.root.pickable_objects
                        if obj.name == pick_set.pickable_object_name
                    ][0]
                    # TODO hardcoded default point size
                    point_size = pickable_object.radius if pickable_object.radius else 50
                    self.viewer.add_points(
                        points,
                        name=f"Picks: {pick_set.meta.pickable_object_name}",
                        size=point_size,
                        face_color=colors,
                        out_of_slice_display=True,
                    )
                    self.info_label.setText(
                        f"Loaded Picks: {pick_set.meta.pickable_object_name}"
                    )
                else:
                    self.info_label.setText(
                        f"No points found for Picks: {pick_set.meta.pickable_object_name}"
                    )
            else:
                self.info_label.setText(
                    f"No pick set found for Picks: {pick_set.meta.pickable_object_name}"
                )
        else:
            self.info_label.setText("No parent run found")

    def get_color(self, pick):
        for obj in self.root.pickable_objects:
            if obj.name == pick.meta.object_name:
                return obj.color
        return "white"

    def get_run(self, name):
        return self.root.get_run(name)

    def open_context_menu(self, position):
        print("Opening context menu")
        item = self.tree_view.itemAt(position)
        if not item:
            return

        if self.is_segmentations_or_picks_item(item):
            context_menu = QMenu(self.tree_view)
            if item.text(0) == "Segmentations":
                run_name = item.parent().parent().text(0)
                run = self.root.get_run(run_name)
                self.show_segmentation_widget(run)
            elif item.text(0) == "Picks":
                run_name = item.parent().text(0)
                run = self.root.get_run(run_name)
                self.show_picks_widget(run)
            context_menu.exec_(self.tree_view.viewport().mapToGlobal(position))

    def is_segmentations_or_picks_item(self, item):
        if item.text(0) == "Segmentations" or item.text(0) == "Picks":
            return True
        return False

    def show_segmentation_widget(self, run):
        widget = QWidget()
        widget.setWindowTitle("Create New Segmentation")

        layout = QFormLayout(widget)
        name_input = QLineEdit(widget)
        name_input.setText("segmentation")
        layout.addRow("Name:", name_input)

        session_input = QSpinBox(widget)
        session_input.setValue(0)
        layout.addRow("Session ID:", session_input)

        user_input = QLineEdit(widget)
        user_input.setText("napariCopick")
        layout.addRow("User ID:", user_input)

        voxel_size_input = QComboBox(widget)
        for voxel_spacing in run.voxel_spacings:
            voxel_size_input.addItem(str(voxel_spacing.meta.voxel_size))
        layout.addRow("Voxel Size:", voxel_size_input)

        create_button = QPushButton("Create", widget)
        create_button.clicked.connect(
            lambda: self.create_segmentation(
                widget,
                run,
                name_input.text(),
                session_input.value(),
                user_input.text(),
                float(voxel_size_input.currentText()),
            )
        )
        layout.addWidget(create_button)

        self.viewer.window.add_dock_widget(widget, area="right")

    def show_picks_widget(self, run):
        widget = QWidget()
        widget.setWindowTitle("Create New Picks")

        layout = QFormLayout(widget)
        object_name_input = QComboBox(widget)
        for obj in self.root.config.pickable_objects:
            object_name_input.addItem(obj.name)
        layout.addRow("Object Name:", object_name_input)

        session_input = QSpinBox(widget)
        session_input.setValue(0)
        layout.addRow("Session ID:", session_input)

        user_input = QLineEdit(widget)
        user_input.setText("napariCopick")
        layout.addRow("User ID:", user_input)

        create_button = QPushButton("Create", widget)
        create_button.clicked.connect(
            lambda: self.create_picks(
                widget,
                run,
                object_name_input.currentText(),
                session_input.value(),
                user_input.text(),
            )
        )
        layout.addWidget(create_button)

        self.viewer.window.add_dock_widget(widget, area="right")

    def create_segmentation(
        self, widget, run, name, session_id, user_id, voxel_size
    ):
        seg = run.new_segmentation(
            voxel_size=voxel_size,
            name=name,
            session_id=str(session_id),
            is_multilabel=True,
            user_id=user_id,
        )

        tomo = zarr.open(run.voxel_spacings[0].tomograms[0].zarr(), "r")[
            "0"
        ]

        shape = tomo.shape
        dtype = np.int32

        # Create an empty Zarr array for the segmentation
        zarr_file = zarr.open(seg.zarr(), mode="w")
        zarr_file.create_dataset(
            "data",
            shape=shape,
            dtype=dtype,
            chunks=(128, 128, 128),
            fill_value=0,
        )

        self.populate_tree()
        widget.close()

    def create_picks(self, widget, run, object_name, session_id, user_id):
        run.new_picks(
            object_name=object_name,
            session_id=str(session_id),
            user_id=user_id,
        )
        self.populate_tree()
        widget.close()

    # ------- CellCanvas Integration Methods -------

    def configure_cellcanvas(self):
        """Open a dialog to configure CellCanvas settings."""
        dialog = CellCanvasServerDialog(
            self,
            run_name=self.cellcanvas_settings['run_name'],
            voxel_spacing=self.cellcanvas_settings['voxel_spacing']
        )
        
        if dialog.exec_():
            # Update settings
            self.cellcanvas_settings = dialog.get_settings()
            self.cellcanvas_client = CellCanvasClient(self.cellcanvas_settings['server_url'])
            self.cellcanvas_status_label.setText(f"CellCanvas: Connected to {self.cellcanvas_settings['server_url']}")
            self.info_label.setText(f"CellCanvas configured for run: {self.cellcanvas_settings['run_name']}")
            
            # Test connection to server
            self.test_server_connection()
    
    def test_server_connection(self):
        """Test the connection to the CellCanvas server."""
        # This is a placeholder for future implementation
        # In the future, you can add actual API endpoint tests here
        try:
            # Example: ping the server or get a simple response
            response = requests.get(f"{self.cellcanvas_settings['server_url']}/ping", timeout=5)
            if response.status_code == 200:
                self.cellcanvas_status_label.setText(f"CellCanvas: Connected (✓)")
                return True
            else:
                self.cellcanvas_status_label.setText(f"CellCanvas: Connection error ({response.status_code})")
                return False
        except Exception as e:
            self.cellcanvas_status_label.setText(f"CellCanvas: Connection failed")
            print(f"Server connection failed: {str(e)}")
            return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Copick Plugin")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the copick config file",
    )
    args = parser.parse_args()

    config_path = None

    viewer = napari.Viewer()
    copick_plugin = CopickPlugin(viewer, config_path=(args.config_path if config_path is None else config_path))
    viewer.window.add_dock_widget(copick_plugin, area="right")
    napari.run()
