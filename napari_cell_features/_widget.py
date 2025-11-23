from qtpy.QtWidgets import QWidget, QVBoxLayout, QComboBox, QLabel, QCheckBox, QGroupBox, QListWidget, QPushButton, QListWidgetItem, QDoubleSpinBox, QHBoxLayout, QGridLayout, QSpinBox
from magicgui.widgets import create_widget
import napari
from ._plotter import extract_time_series, PopupPlot, MplCanvas
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from qtpy.QtWidgets import QFileDialog
import pandas as pd
import os


class CellFeaturesWidget(QWidget):
    def __init__(self, napari_viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = napari_viewer
        self.popup = None
        self.selected_labels = set()
        self.centroids_layer = None
        
        self.setLayout(QVBoxLayout())
        
        # Layer selection
        self.image_layer_combo = QComboBox()
        self.labels_layer_combo = QComboBox()
        self.refresh_layers()
        
        self.layout().addWidget(QLabel("Image Layer:"))
        self.layout().addWidget(self.image_layer_combo)
        self.layout().addWidget(QLabel("Labels Layer:"))
        self.layout().addWidget(self.labels_layer_combo)
        
        # Plot options
        self.options_group = QGroupBox("Plot Options")
        options_layout = QGridLayout()
        
        # Row 0: Show Markers + Linewidth
        self.show_markers_check = QCheckBox("Show Markers")
        self.show_markers_check.setChecked(True)
        self.show_markers_check.stateChanged.connect(self.update_plot)
        options_layout.addWidget(self.show_markers_check, 0, 0)
        
        lw_layout = QHBoxLayout()
        lw_layout.addWidget(QLabel("Linewidth:"))
        self.linewidth_spin = QDoubleSpinBox()
        self.linewidth_spin.setRange(0.5, 10.0)
        self.linewidth_spin.setValue(1.5)
        self.linewidth_spin.setSingleStep(0.5)
        self.linewidth_spin.valueChanged.connect(self.update_plot)
        lw_layout.addWidget(self.linewidth_spin)
        options_layout.addLayout(lw_layout, 0, 1)
        
        # Row 1: Normalize + Smooth
        self.normalize_check = QCheckBox("Normalize")
        self.normalize_check.setChecked(False)
        self.normalize_check.stateChanged.connect(self.update_plot)
        options_layout.addWidget(self.normalize_check, 1, 0)
        
        smooth_layout = QHBoxLayout()
        self.smooth_check = QCheckBox("Smooth")
        self.smooth_check.stateChanged.connect(self.update_plot)
        smooth_layout.addWidget(self.smooth_check)
        
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(2, 50)
        self.smooth_spin.setValue(5)
        self.smooth_spin.valueChanged.connect(self.update_plot)
        smooth_layout.addWidget(self.smooth_spin)
        options_layout.addLayout(smooth_layout, 1, 1)
        
        self.options_group.setLayout(options_layout)
        self.layout().addWidget(self.options_group)
        
        self.select_all_btn = QPushButton("Select All Labels")
        self.select_all_btn.clicked.connect(self.select_all)
        self.layout().addWidget(self.select_all_btn)
        
        self.clear_sel_btn = QPushButton("Clear Selection")
        self.clear_sel_btn.clicked.connect(self.clear_selection)
        self.layout().addWidget(self.clear_sel_btn)

        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_csv)
        self.layout().addWidget(self.export_btn)

        # In-widget plot canvas
        self.widget_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.layout().addWidget(self.widget_canvas)
        
        # Connect pick event
        self.widget_canvas.mpl_connect('pick_event', self.on_pick)
        
        # Events
        self.viewer.layers.events.inserted.connect(self.refresh_layers)
        self.viewer.layers.events.removed.connect(self.refresh_layers)
        self.viewer.dims.events.current_step.connect(self.update_time_line)
        
        # Mouse callback
        self.viewer.mouse_drag_callbacks.append(self._on_click)

    def on_pick(self, event):
        artist = event.artist
        label = getattr(artist, 'label_id', None)
        if label is not None and label in self.selected_labels:
            self.selected_labels.remove(label)
            self.update_plot()

    def update_time_line(self, event=None):
        if hasattr(self, 'vline') and self.vline:
            current_time = self.viewer.dims.current_step[0]
            self.vline.set_xdata([current_time, current_time])
            self.widget_canvas.draw()
        self.update_centroids()


    def refresh_layers(self, event=None):
        self.image_layer_combo.clear()
        self.labels_layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self.image_layer_combo.addItem(layer.name)
            elif isinstance(layer, napari.layers.Labels):
                self.labels_layer_combo.addItem(layer.name)

    def select_all(self):
        lbl_name = self.labels_layer_combo.currentText()
        if not lbl_name:
            return
        lbl_layer = self.viewer.layers[lbl_name]
        
        # Get all unique labels (excluding 0)
        # This might be slow for very large data
        unique_labels = np.unique(lbl_layer.data)
        unique_labels = unique_labels[unique_labels != 0]
        
        self.selected_labels.update(unique_labels)
        self.update_plot()

    def clear_selection(self):
        self.selected_labels.clear()
        self.update_plot()
        self.update_centroids()

    def update_centroids(self):
        lbl_name = self.labels_layer_combo.currentText()
        if not lbl_name:
            return
        lbl_layer = self.viewer.layers[lbl_name]
        
        current_time = self.viewer.dims.current_step[0]
        
        centroids = []
        
        # Check if 2D+t (3D) or just 2D
        # Assuming (T, Y, X) for now based on extract_time_series
        
        if lbl_layer.data.ndim == 3:
            current_labels = lbl_layer.data[current_time]
            
            for label_id in self.selected_labels:
                mask = current_labels == label_id
                if np.any(mask):
                    # center_of_mass returns (y, x)
                    cm = center_of_mass(mask)
                    # Napari points are (y, x) in 2D view, but we are in 3D (T, Y, X) world usually?
                    # If we add points to a layer, we need to match the layer dims.
                    # If we want points to only show up at current time, we can add them as (T, Y, X)
                    # or manage a 2D points layer that we update every time step.
                    # Updating a 2D points layer is often smoother for "marker" behavior.
                    centroids.append(cm)
        
        # Update or create points layer
        if self.centroids_layer is None or self.centroids_layer not in self.viewer.layers:
            if centroids:
                self.centroids_layer = self.viewer.add_points(
                    np.array(centroids),
                    name='Selected Centroids',
                    face_color='white',
                    symbol='ring',
                    size=10,
                    n_dimensional=False
                )
                self.centroids_layer.edge_color = 'white'
        else:
            if centroids:
                self.centroids_layer.data = np.array(centroids)
            else:
                self.centroids_layer.data = np.empty((0, 2))
                
    def export_csv(self):
        if not self.selected_labels:
            return
            
        img_name = self.image_layer_combo.currentText()
        lbl_name = self.labels_layer_combo.currentText()
        
        if not img_name or not lbl_name:
            return
            
        img_layer = self.viewer.layers[img_name]
        lbl_layer = self.viewer.layers[lbl_name]
        
        # Ask for folder
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return
            
        # Prepare data
        data_list = []
        for val in self.selected_labels:
            times, intensities = extract_time_series(img_layer, lbl_layer, val)
            # Create a dataframe for this label? Or one big CSV?
            # "save the selected plotted lines in a csv file that has the same default name as the image"
            # Implies one file.
            
            # Let's make a DataFrame with Time, Label, Intensity
            df = pd.DataFrame({
                'Time': times,
                'Label': val,
                'Intensity': intensities
            })
            data_list.append(df)
            
        if not data_list:
            return
            
        final_df = pd.concat(data_list, ignore_index=True)
        
        # Pivot to have Time as index and Labels as columns? 
        # Or just long format? "selected plotted lines" -> usually users want Time vs Label columns.
        # Let's pivot.
        pivot_df = final_df.pivot(index='Time', columns='Label', values='Intensity')
        
        # Filename
        filename = f"{img_name}.csv"
        filepath = os.path.join(folder, filename)
        
        pivot_df.to_csv(filepath)
        print(f"Exported to {filepath}")



    def _on_click(self, viewer, event):
        if event.button != 1:
            return
            
        # Check if we are in a valid state
        img_name = self.image_layer_combo.currentText()
        lbl_name = self.labels_layer_combo.currentText()
        
        if not img_name or not lbl_name:
            return
            
        lbl_layer = self.viewer.layers[lbl_name]
        
        # Get coordinates
        cursor_pos = lbl_layer.world_to_data(event.position)
        cursor_index = tuple(int(round(x)) for x in cursor_pos)
        
        try:
            val = lbl_layer.data[cursor_index]
        except IndexError:
            return 
            
        if val == 0:
            return
            
        # Toggle selection
        if val in self.selected_labels:
            self.selected_labels.remove(val)
        else:
            self.selected_labels.add(val)
            
        self.update_plot()
        self.update_centroids()

    def update_plot(self):
        img_name = self.image_layer_combo.currentText()
        lbl_name = self.labels_layer_combo.currentText()
        
        if not img_name or not lbl_name:
            return
            
        img_layer = self.viewer.layers[img_name]
        lbl_layer = self.viewer.layers[lbl_name]

        self.widget_canvas.axes.clear()
        self.widget_canvas.axes.patch.set_alpha(0)
        
        # Get options
        show_markers = self.show_markers_check.isChecked()
        linewidth = self.linewidth_spin.value()
        normalize = self.normalize_check.isChecked()
        smooth = self.smooth_check.isChecked()
        smooth_window = self.smooth_spin.value()
        
        marker = 'o' if show_markers else None
        markersize = linewidth * 3 # Adapt marker size
        
        # Plot all selected labels
        for val in self.selected_labels:
            times, intensities = extract_time_series(img_layer, lbl_layer, val)
            
            # Smoothing
            if smooth and len(intensities) > smooth_window:
                # Simple moving average
                window = np.ones(smooth_window) / smooth_window
                # Use 'valid' to avoid boundary effects
                intensities = np.convolve(intensities, window, mode='valid')
                # Adjust times to match the valid range
                # The convolution result is centered.
                # For window size W, we lose (W-1) points total.
                # Start offset is roughly (W-1)//2
                offset = (smooth_window - 1) // 2
                times = list(times) # Ensure it's a list/array
                times = times[offset : offset + len(intensities)]

            if normalize:
                mean_val = np.nanmean(intensities)
                if mean_val != 0:
                    intensities = [i / mean_val for i in intensities]

            
            try:
                color = lbl_layer.get_color(val)
                if color is None:
                    color = 'white'
            except AttributeError:
                color = 'white'
                
            line, = self.widget_canvas.axes.plot(times, intensities, marker=marker, color=color, label=f"Label {val}", picker=5, linewidth=linewidth, markersize=markersize)
            line.label_id = val

        self.widget_canvas.axes.set_title(f"Intensity over Time", color='white')
        self.widget_canvas.axes.set_xlabel("Time", color='white')
        ylabel = "Normalized Intensity" if normalize else "Mean Intensity"
        self.widget_canvas.axes.set_ylabel(ylabel, color='white')
        self.widget_canvas.axes.tick_params(axis='x', colors='white')
        self.widget_canvas.axes.tick_params(axis='y', colors='white')
        
        # Current time line
        current_time = self.viewer.dims.current_step[0]
        self.vline = self.widget_canvas.axes.axvline(x=current_time, color='white', linestyle='--')
        
        # Set spines to white
        for spine in self.widget_canvas.axes.spines.values():
            spine.set_color('white')
            
        self.widget_canvas.draw()




