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
# requires-python = ">=3.11"
# dependencies = [
#     "napari>=0.9,<1",
#     "PySide6>=6.7,<7",
#     "copick>=1.31,<2",
#     "zarr<3",
#     "fsspec[http]",
#     "requests",
#     "numpy",
#     "superqt>=0.7",
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
    QFormLayout,
    QComboBox,
    QSpinBox,
    QGroupBox,
    QHBoxLayout,
    QCheckBox,
    QGridLayout,
    QDialog,
)
from qtpy.QtCore import Qt, QPoint, QEvent, Slot
from cellcanvas_painting import PaintingSync
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
        self.connected = False
        self.last_error = None
    
    def send_request(self, endpoint, params=None, method="GET"):
        """Send a request to the server."""
        url = f"{self.server_url}/{endpoint}"
        print(f"Sending {method} request to {url}")
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, json=params, timeout=10)
            else:
                error_msg = f"Unsupported method: {method}"
                print(error_msg)
                self.last_error = error_msg
                self.connected = False
                return None
                
            if response.status_code != 200:
                error_msg = f"Error {response.status_code}: {response.text}"
                print(error_msg)
                self.last_error = error_msg
                self.connected = False
                return None
                
            # Request succeeded, mark as connected
            self.connected = True
            self.last_error = None
            return response.json()
        except requests.exceptions.ConnectionError:
            error_msg = f"Connection error: Could not connect to {url}"
            print(error_msg)
            self.last_error = error_msg
            self.connected = False
            return None
        except requests.exceptions.Timeout:
            error_msg = f"Timeout error: Request to {url} timed out"
            print(error_msg)
            self.last_error = error_msg
            self.connected = False
            return None
        except Exception as e:
            error_msg = f"Request error: {str(e)}"
            print(error_msg)
            self.last_error = error_msg
            self.connected = False
            return None
    
    def test_connection(self):
        """Test the connection to the server."""
        try:
            response = self.send_request("ping")
            if response and response.get("status") == "ok":
                self.connected = True
                return True
            else:
                self.connected = False
                return False
        except Exception as e:
            print(f"Connection test failed: {str(e)}")
            self.connected = False
            self.last_error = str(e)
            return False
    
    def get_status_text(self):
        """Get a status text for display."""
        if self.connected:
            return f"CellCanvas: Connected to {self.server_url} (✓)"
        else:
            error_text = f" - {self.last_error}" if self.last_error else ""
            return f"CellCanvas: Not connected{error_text}"

from qtpy.QtCore import QObject, Signal

# should use psygnal
class TreeUpdateSignaler(QObject):
    """Simple signaler for updating tree items from background threads."""
    update_tree = Signal(object, object)  # parent_item, data_callback

# Add this global signaler
_tree_signaler = TreeUpdateSignaler()

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
        
        # For tracking active tomogram and segmentations
        self.active_tomogram = None
        self.active_tomogram_scale = None
        self.active_painting_layer = None
        self.active_prediction_layer = None
        self.current_run_name = None
        self.current_voxel_size = None
        self.painting_sync = None
        self.painting_layer = None
        self.prediction_layer = None
        
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
        self.painting_status_label = QLabel("No remote painting loaded.")
        self.painting_status_label.setWordWrap(True)
        cellcanvas_layout.addWidget(self.painting_status_label)
        self.retry_painting_button = QPushButton("Retry Painting Save")
        self.retry_painting_button.setEnabled(False)
        self.retry_painting_button.clicked.connect(self.retry_painting)
        cellcanvas_layout.addWidget(self.retry_painting_button)
        self.save_painting_copy_button = QPushButton("Save Local Copy and Disconnect…")
        self.save_painting_copy_button.setEnabled(False)
        self.save_painting_copy_button.clicked.connect(self.save_painting_copy)
        cellcanvas_layout.addWidget(self.save_painting_copy_button)
        
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
        if not self._can_switch_painting():
            return False
        if path:
            root = copick.from_file(path)
            self._detach_painting()
            self.root = root
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
            # Just load the main categories, not the actual data
            self.create_run_structure(item, data)
        elif isinstance(data, copick.models.CopickVoxelSpacing):
            self.create_voxel_spacing_structure(item, data)
        elif item.text(0) == "Picks":
            # Lazy load picks when the Picks item is expanded
            self.expand_picks(item, item.parent().data(0, Qt.UserRole))
        elif item.text(0) == "Tomograms":
            # Lazy load tomograms
            self.expand_tomograms(item, item.parent().data(0, Qt.UserRole))
        elif item.text(0) == "Segmentations":
            # Lazy load segmentations
            self.expand_segmentations(item, item.parent().data(0, Qt.UserRole))

    def create_run_structure(self, item, run):
        """Create just the structure without loading all data."""
        if not item.childCount():
            # Add voxel spacing items
            for voxel_spacing in run.voxel_spacings:
                spacing_item = QTreeWidgetItem(
                    item, [f"Voxel Spacing: {voxel_spacing.meta.voxel_size}"]
                )
                spacing_item.setData(0, Qt.UserRole, voxel_spacing)
                spacing_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
            
            # Just add the Picks category, not the actual data
            picks_item = QTreeWidgetItem(item, ["Picks"])
            picks_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
            
            item.addChild(picks_item)

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

    def expand_picks(self, item, run):
        """Lazily load picks."""
        if not item.childCount():
            # Load picks directly (no threading)
            user_dict = {}
            for pick in run.picks:
                if pick.meta.user_id not in user_dict:
                    user_dict[pick.meta.user_id] = {}
                if pick.meta.session_id not in user_dict[pick.meta.user_id]:
                    user_dict[pick.meta.user_id][pick.meta.session_id] = []
                user_dict[pick.meta.user_id][pick.meta.session_id].append(pick)
            
            for user_id, sessions in user_dict.items():
                user_item = QTreeWidgetItem(item, [f"User: {user_id}"])
                for session_id, picks in sessions.items():
                    session_item = QTreeWidgetItem(user_item, [f"Session: {session_id}"])
                    for pick in picks:
                        pick_child = QTreeWidgetItem(session_item, [pick.meta.pickable_object_name])
                        pick_child.setData(0, Qt.UserRole, pick)

    # Helper method to safely update UI from background thread
    def update_ui_picks(self, callback):
        callback()

    def create_voxel_spacing_structure(self, item, voxel_spacing):
        """Create just the structure of voxel spacing items."""
        if not item.childCount():
            # Just add category items, not the actual data
            tomogram_item = QTreeWidgetItem(item, ["Tomograms"])
            tomogram_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
            item.addChild(tomogram_item)
            
            segmentation_item = QTreeWidgetItem(item, ["Segmentations"])
            segmentation_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
            item.addChild(segmentation_item)
    
    def expand_tomograms(self, item, voxel_spacing):
        """Lazily load tomograms."""
        if not item.childCount():
            # Load tomograms directly (no threading)
            tomograms = voxel_spacing.tomograms
            for tomogram in tomograms:
                tomo_child = QTreeWidgetItem(item, [tomogram.meta.tomo_type])
                tomo_child.setData(0, Qt.UserRole, tomogram)
    
    # Helper method to safely update UI from background thread
    def update_ui_tomograms(self, callback):
        callback()
    
    def expand_segmentations(self, item, voxel_spacing):
        """Lazily load segmentations."""
        if not item.childCount():
            # Load segmentations directly (no threading)
            segmentations = voxel_spacing.run.get_segmentations(
                voxel_size=voxel_spacing.meta.voxel_size
            )
            for segmentation in segmentations:
                seg_child = QTreeWidgetItem(item, [segmentation.meta.name])
                seg_child.setData(0, Qt.UserRole, segmentation)
    
    # Helper method to safely update UI from background thread
    def update_ui_segmentations(self, callback):
        callback()

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
        if not self._can_switch_painting():
            return None
        # TODO update cell canvas settings when tomogram is loaded
        # self.cellcanvas_settings['run_name'] = data.meta.name
        # self.cellcanvas_settings['voxel_spacing'] = data.meta.voxel_size

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
        run_name = None
        for run in self.root.runs:
            for voxel_spacing in run.voxel_spacings:
                if tomogram in voxel_spacing.tomograms:
                    voxel_size = voxel_spacing.meta.voxel_size
                    run_name = run.meta.name
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
        
        # Update active tomogram properties
        self.active_tomogram = tomogram
        
        # Automatically load or create segmentations from the CellCanvas server
        if run_name and voxel_size:
            # Ensure segmentations exist on the server
            self.ensure_and_load_segmentations(run_name, voxel_size, tomogram.meta.tomo_type)

        self.info_label.setText(
            f"Loaded Tomogram: {tomogram.meta.tomo_type} with num scales = {len(scale_levels)}"
        )

        return layer

    def ensure_and_load_segmentations(self, run_name, voxel_size, tomogram_type):
        """Ensure painting and prediction segmentations exist and load them from the server via HTTP zarr."""
        if not self._can_switch_painting():
            return False
        print(f"Ensuring and loading segmentations for {run_name}, voxel size {voxel_size}, tomogram {tomogram_type}")
        
        # First, check if the server is connected
        if not self.cellcanvas_client.test_connection():
            print("Warning: Server not connected. Attempting to reconnect...")
            # Try to reconnect
            if not self.cellcanvas_client.test_connection():
                print("Error: Cannot connect to CellCanvas server. Segmentations will not be synchronized.")
                return False
        
        # Prepare the request data
        request_data = {
            'run_name': run_name,
            'voxel_size': voxel_size,
            'tomogram_type': tomogram_type,
            'user_id': 'napariUser',  # Default user ID
            'session_id': self.session_id,
            'painting_name': 'painting',
            'prediction_name': 'prediction'
        }
        
        # Send the request to ensure segmentations exist
        response = self.cellcanvas_client.send_request(
            'api/segmentation/ensure',
            params=request_data,
            method='POST'
        )
        
        if response and response.get('status') == 'success':
            print("Successfully ensured segmentations exist on server")
            
            # Load the segmentations via HTTP zarr
            server_url = self.cellcanvas_client.server_url
            base_url = server_url.rstrip("/")
            
            # Construct URLs for painting and prediction segmentations
            painting_url = f"{base_url}/{run_name}/Segmentations/{voxel_size}_napariUser_{self.session_id}_painting-multilabel.zarr"
            prediction_url = f"{base_url}/{run_name}/Segmentations/{voxel_size}_napariUser_{self.session_id}_prediction-multilabel.zarr"
            
            print(f"Loading painting segmentation from: {painting_url}")
            painting = self._load_segmentation_from_url(painting_url, "Painting", voxel_size)
            
            print(f"Loading prediction segmentation from: {prediction_url}")
            prediction = self._load_segmentation_from_url(prediction_url, "Prediction", voxel_size)
            if painting is None or prediction is None:
                for layer in (painting, prediction):
                    if layer is not None:
                        self.viewer.layers.remove(layer)
                self.info_label.setText("Could not load remote annotations; previous painting is still attached.")
                return False

            self.current_run_name, self.current_voxel_size = run_name, voxel_size
            self.painting_layer, self.prediction_layer = painting, prediction
            
            # Set up event listener for painting updates if the layer was loaded
            if self.painting_layer:
                self.active_painting_layer = self.painting_layer
                self.setup_painting_listener(self.painting_layer)
                print(f"Successfully set up painting layer with event listener")
            
            # Store a reference to the prediction layer if it was loaded
            if self.prediction_layer:
                self.active_prediction_layer = self.prediction_layer
            
            return True
        else:
            print("Failed to ensure segmentations exist on server")
            if response and 'error' in response:
                print(f"Error: {response['error']}")
            
        return False

    def _load_segmentation_from_url(self, url, name_prefix, voxel_size):
        """Load a segmentation from a zarr URL into napari."""
        layer = None
        try:
            import zarr
            from fsspec import get_mapper
            
            print(f"Loading {name_prefix} segmentation from URL: {url}")
            
            # Open the zarr store using fsspec
            store = get_mapper(url)
            zarr_root = zarr.open(store, mode="r")
            
            # Find the data array - either "data" or "0" are common dataset names
            data_arr = None
            if "data" in zarr_root:
                data_arr = zarr_root["data"]
                print(f"Found 'data' dataset in zarr store")
            elif "0" in zarr_root:
                data_arr = zarr_root["0"]
                print(f"Found '0' dataset in zarr store")
            else:
                # Try to find any array in the store
                for key in zarr_root.keys():
                    if isinstance(zarr_root[key], zarr.core.Array):
                        data_arr = zarr_root[key]
                        print(f"Found '{key}' dataset in zarr store")
                        break
            
            if data_arr is None:
                print(f"Error: No data array found in {url}")
                return None
            
            print(f"Loading array with shape {data_arr.shape} and dtype {data_arr.dtype}")
            
            # Create scale array based on voxel size
            scale = [float(voxel_size)] * 3
            
            # Add the segmentation to napari
            layer = self.viewer.add_labels(
                data_arr[:] if name_prefix.lower() == "painting" else data_arr,
                name=f"{name_prefix} Segmentation",
                scale=scale,
                opacity=0.5
            )
            
            if name_prefix.lower() != "painting":
                layer.editable = False

            # For painting layer, set up the colormap and painting labels
            if name_prefix.lower() == "painting" and hasattr(self.root, 'config') and hasattr(self.root.config, 'pickable_objects'):
                # Create a color map based on copick colors
                colormap = self.get_copick_colormap()
                
                # Apply the colormap
                from napari.utils import DirectLabelColormap
                layer.colormap = DirectLabelColormap(color_dict=colormap)
                
                # Set up the painting labels
                layer.painting_labels = [
                    obj.label for obj in self.root.config.pickable_objects
                ]
                
                # Store the class labels mapping
                self.class_labels_mapping = {
                    obj.label: obj.name for obj in self.root.config.pickable_objects
                }
            
            print(f"Successfully loaded {name_prefix} segmentation")
            return layer
            
        except Exception as e:
            if layer is not None:
                self.viewer.layers.remove(layer)
            print(f"Error loading {name_prefix} segmentation from {url}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def setup_painting_listener(self, layer):
        """Bind an editable local buffer to a fixed server/project context."""
        self._detach_painting()
        context = {
            "run_name": self.current_run_name, "voxel_size": self.current_voxel_size,
            "user_id": "napariUser", "session_id": str(self.session_id),
            "segmentation_name": "painting",
        }
        client = CellCanvasClient(self.cellcanvas_client.server_url)
        self.painting_sync = PaintingSync(layer, client, context, self)
        self.painting_sync.status_changed.connect(self._painting_status)
        self._painting_status("Painting saved.")

    @Slot(str)
    def _painting_status(self, message):
        self.painting_status_label.setText(message)
        self.retry_painting_button.setEnabled(
            self.painting_sync is not None and self.painting_sync.failed)
        self.save_painting_copy_button.setEnabled(self.painting_sync is not None)

    @Slot()
    def retry_painting(self):
        if self.painting_sync is not None:
            self.painting_sync.retry()

    @Slot()
    def save_painting_copy(self):
        if self.painting_sync is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save local ZYX painting", "painting.npy",
                                            "NumPy array (*.npy)")
        if not path:
            return
        try:
            np.save(path, self.painting_sync.layer.data, allow_pickle=False)
        except Exception as error:
            self.info_label.setText(f"Could not save local painting: {error}")
            return
        self.painting_sync.detach(saved_locally=True)
        self.painting_sync.deleteLater()
        self.painting_sync = None
        self._painting_status("Local copy saved; remote sync disconnected. An in-flight request may still finish.")

    def _can_switch_painting(self):
        if self.painting_sync is not None and self.painting_sync.pending:
            self.info_label.setText("Retry painting or save a local copy before switching or closing.")
            return False
        return True

    def _detach_painting(self):
        if self.painting_sync is not None:
            self.painting_sync.detach()
            self.painting_sync.deleteLater()
            self.painting_sync = None
        self.retry_painting_button.setEnabled(False)
        self.save_painting_copy_button.setEnabled(False)

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
        labels_layer = self.viewer.add_labels(
            data,
            name=f"Segmentation: {segmentation.meta.name}",
            scale=scale,
            opacity=0.5  # Add some transparency to make it easier to see
        )
        
        # Apply the colormap
        labels_layer.colormap = DirectLabelColormap(color_dict=colormap)
        
        # Set up the painting labels
        labels_layer.painting_labels = [
            obj.label for obj in self.root.config.pickable_objects
        ]
        
        # Store the class labels mapping
        self.class_labels_mapping = {
            obj.label: obj.name for obj in self.root.config.pickable_objects
        }

        self.info_label.setText(f"Loaded Segmentation: {segmentation.meta.name}")
        
        # Return the layer so we can keep track of it
        return labels_layer

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
        
        if not self._can_switch_painting():
            dialog.deleteLater()
            return
        if dialog.exec_():
            self._detach_painting()
            # Update settings
            self.cellcanvas_settings = dialog.get_settings()
            self.cellcanvas_client = CellCanvasClient(self.cellcanvas_settings['server_url'])
            
            # Test connection to server
            if self.cellcanvas_client.test_connection():
                self.cellcanvas_status_label.setText(f"CellCanvas: Connected to {self.cellcanvas_settings['server_url']} (✓)")
                self.info_label.setText(f"CellCanvas configured for run: {self.cellcanvas_settings['run_name']}")
            else:
                self.cellcanvas_status_label.setText(self.cellcanvas_client.get_status_text())
                self.info_label.setText("Warning: Could not connect to CellCanvas server. Check settings.")
        dialog.deleteLater()

    def test_server_connection(self):
        """Test the connection to the CellCanvas server."""
        if self.cellcanvas_client.test_connection():
            self.cellcanvas_status_label.setText(self.cellcanvas_client.get_status_text())
            return True
        else:
            self.cellcanvas_status_label.setText(self.cellcanvas_client.get_status_text())
            return False

    def cleanup(self):
        """Detach a saved buffer without discarding outstanding edits."""
        if not self._can_switch_painting():
            return False
        self._detach_painting()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        return True

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Close and not self._can_switch_painting():
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event):
        if not self.cleanup():
            event.ignore()
            return
        super().closeEvent(event)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CellCanvas Napari-Copick Client")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the copick config file"
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:8000",
        help="URL of the CellCanvas server"
    )
    args = parser.parse_args()

    viewer = napari.Viewer()
    copick_plugin = CopickPlugin(viewer, config_path=args.config_path)
    
    # Set the server URL from command line
    copick_plugin.cellcanvas_settings['server_url'] = args.server_url
    copick_plugin.cellcanvas_client = CellCanvasClient(args.server_url)
    copick_plugin.cellcanvas_status_label.setText(f"CellCanvas: Connected to {args.server_url}")
    
    viewer.window.add_dock_widget(copick_plugin, area="right")
    copick_plugin.window().installEventFilter(copick_plugin)
    try:
        napari.run()
    finally:
        copick_plugin.cleanup()
