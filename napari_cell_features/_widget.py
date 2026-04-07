from qtpy.QtWidgets import QWidget, QVBoxLayout, QComboBox, QLabel, QCheckBox, QGroupBox, QListWidget, QPushButton, QListWidgetItem, QDoubleSpinBox, QHBoxLayout, QGridLayout, QSpinBox, QTabWidget, QScrollArea, QFileDialog
from qtpy.QtCore import QTimer
import napari
from ._plotter import extract_time_series, PopupPlot, MplCanvas
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, shift, gaussian_filter, binary_erosion
from scipy.signal import correlate
from scipy.optimize import least_squares
import pandas as pd
import os


def robust_linear_fit(x, y, seed=42):
    """
    Perform robust linear regression with RANSAC-inspired initialization.
    This avoids being pulled by outliers (like cell points in the background).
    """
    if len(x) < 2:
        return 0.0, 0.0

    # Cast to float to avoid overflow issues with uint16 data
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Use a local random state for determinism
    rng = np.random.RandomState(seed)

    # 1. RANSAC-like initialization
    # Take random pairs of points to find the most representative line
    n_pts = len(x)
    sub_size = min(n_pts, 500)
    if n_pts > sub_size:
        indices = rng.choice(n_pts, sub_size, replace=False)
        xs, ys = x[indices], y[indices]
    else:
        xs, ys = x, y

    best_a, best_b = 0.0, 0.0
    max_inliers = -1

    # Try 50 random pairs
    for _ in range(50):
        pair_idx = rng.choice(len(xs), 2, replace=False)
        x1, y1 = xs[pair_idx[0]], ys[pair_idx[0]]
        x2, y2 = xs[pair_idx[1]], ys[pair_idx[1]]

        if x1 == x2:
            continue

        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1

        # Count inliers on the subsample
        # Use 5% of the Y range as a rough threshold for "agreement"
        y_range = np.ptp(ys)
        threshold = max(y_range * 0.05, 1.0)

        residuals = np.abs(a * xs + b - ys)
        inliers = np.sum(residuals < threshold)

        if inliers > max_inliers:
            max_inliers = inliers
            best_a, best_b = a, b

    # 2. Refine with least_squares (Soft L1 loss) starting from the best consensus
    def model(params, x):
        return params[0] * x + params[1]

    def residuals_fn(params, x, y):
        return model(params, x) - y

    # Estimate noise scale from the consensus residuals
    consensus_res = np.abs(best_a * xs + best_b - ys)
    f_scale = np.median(consensus_res) * 2.0
    if f_scale <= 0:
        f_scale = 1.0

    res = least_squares(residuals_fn, [best_a, best_b], args=(x, y), loss='soft_l1', f_scale=f_scale)
    return res.x


class CellFeaturesWidget(QWidget):
    def __init__(self, napari_viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = napari_viewer
        self.popup = None
        self.selected_labels = set()
        self.centroids_layer = None
        self.last_auto_shift = (0.0, 0.0)

        # Debounce timer for refresh_layers
        self._refresh_timer = QTimer()
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(200) # ms
        self._refresh_timer.timeout.connect(self._do_refresh_layers)

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Scroll Area Setup
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        main_layout.addWidget(self.scroll)

        self.container = QWidget()
        self.container_layout = QVBoxLayout()
        self.container.setLayout(self.container_layout)
        self.scroll.setWidget(self.container)

        # Tab Widget
        self.tabs = QTabWidget()
        self.container_layout.addWidget(self.tabs)

        # --- TAB 1: TIME SERIES ---
        self.time_series_tab = QWidget()
        self.time_series_tab.setLayout(QVBoxLayout())
        self.tabs.addTab(self.time_series_tab, "Time Series")

        # Layer selection
        self.image_layer_combo = QComboBox()
        self.labels_layer_combo = QComboBox()

        self.time_series_tab.layout().addWidget(QLabel("Image Layer:"))
        self.time_series_tab.layout().addWidget(self.image_layer_combo)
        self.time_series_tab.layout().addWidget(QLabel("Labels Layer:"))
        self.time_series_tab.layout().addWidget(self.labels_layer_combo)

        # Plot options
        self.options_group = QGroupBox("Plot Options")
        options_layout = QGridLayout()
        self.options_group.setLayout(options_layout)

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

        self.show_avg_check = QCheckBox("Show Average of All")
        self.show_avg_check.setChecked(False)
        self.show_avg_check.stateChanged.connect(self.update_plot)
        options_layout.addWidget(self.show_avg_check, 2, 0)

        self.show_bg_ts_check = QCheckBox("Show Background")
        self.show_bg_ts_check.setChecked(False)
        self.show_bg_ts_check.stateChanged.connect(self.update_plot)
        options_layout.addWidget(self.show_bg_ts_check, 2, 1)

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

        self.time_series_tab.layout().addWidget(self.options_group)

        self.select_all_btn = QPushButton("Select All Labels")
        self.select_all_btn.clicked.connect(self.select_all)
        self.time_series_tab.layout().addWidget(self.select_all_btn)

        self.clear_sel_btn = QPushButton("Clear Selection")
        self.clear_sel_btn.clicked.connect(self.clear_selection)
        self.time_series_tab.layout().addWidget(self.clear_sel_btn)

        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_csv)
        self.time_series_tab.layout().addWidget(self.export_btn)

        # In-widget plot canvas
        self.widget_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.time_series_tab.layout().addWidget(self.widget_canvas)

        # --- TAB 2: SCATTER PLOT ---
        self.scatter_tab = QWidget()
        self.scatter_tab.setLayout(QVBoxLayout())
        self.tabs.addTab(self.scatter_tab, "Scatter Plot")

        self.image_x_combo = QComboBox()
        self.image_y_combo = QComboBox()
        self.image_x_combo.currentIndexChanged.connect(self.update_scatter_plot)
        self.image_y_combo.currentIndexChanged.connect(self.update_scatter_plot)

        self.scatter_tab.layout().addWidget(QLabel("Image X (e.g. 405):"))
        self.scatter_tab.layout().addWidget(self.image_x_combo)
        self.scatter_tab.layout().addWidget(QLabel("Image Y (e.g. 491):"))
        self.scatter_tab.layout().addWidget(self.image_y_combo)

        # Scatter options group
        scatter_options = QGroupBox("Scatter Options")
        scatter_opt_layout = QGridLayout()
        scatter_options.setLayout(scatter_opt_layout)

        # Row 0
        scatter_opt_layout.addWidget(QLabel("Alpha:"), 0, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.01, 1.0)
        self.alpha_spin.setValue(0.1)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.valueChanged.connect(self.update_scatter_plot)
        scatter_opt_layout.addWidget(self.alpha_spin, 0, 1)

        scatter_opt_layout.addWidget(QLabel("Size:"), 0, 2)
        self.psize_spin = QSpinBox()
        self.psize_spin.setRange(1, 100)
        self.psize_spin.setValue(2)
        self.psize_spin.valueChanged.connect(self.update_scatter_plot)
        scatter_opt_layout.addWidget(self.psize_spin, 0, 3)

        # Row 1
        self.show_bg_check = QCheckBox("Show BG")
        self.show_bg_check.stateChanged.connect(self.update_scatter_plot)
        scatter_opt_layout.addWidget(self.show_bg_check, 1, 0, 1, 2)

        self.fit_bg_check = QCheckBox("Fit BG")
        self.fit_bg_check.stateChanged.connect(self.update_scatter_plot)
        scatter_opt_layout.addWidget(self.fit_bg_check, 1, 2, 1, 2)

        # Row 2
        self.perform_fit_check = QCheckBox("Fit Cell")
        self.perform_fit_check.stateChanged.connect(self.update_scatter_plot)
        scatter_opt_layout.addWidget(self.perform_fit_check, 2, 0, 1, 2)

        scatter_opt_layout.addWidget(QLabel("Fit %tile:"), 2, 2)
        self.fit_percentile_spin = QSpinBox()
        self.fit_percentile_spin.setRange(0, 100)
        self.fit_percentile_spin.setValue(50)
        self.fit_percentile_spin.setMinimumWidth(50)
        self.fit_percentile_spin.valueChanged.connect(self.update_scatter_plot)
        scatter_opt_layout.addWidget(self.fit_percentile_spin, 2, 3)

        # Row 3: Contour Toggle
        self.scatter_on_contours_check = QCheckBox("Calculate on Contours Only")
        self.scatter_on_contours_check.stateChanged.connect(self.update_scatter_plot)
        scatter_opt_layout.addWidget(self.scatter_on_contours_check, 3, 0, 1, 4)

        self.scatter_tab.layout().addWidget(scatter_options)

        self.export_scatter_btn = QPushButton("Export Scatter CSV")
        self.export_scatter_btn.clicked.connect(self.export_scatter_csv)
        self.scatter_tab.layout().addWidget(self.export_scatter_btn)

        self.scatter_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.scatter_tab.layout().addWidget(self.scatter_canvas)

        # --- TAB 3: IMAGE RATIO ---
        self.ratio_tab = QWidget()
        self.ratio_tab.setLayout(QVBoxLayout())
        self.tabs.addTab(self.ratio_tab, "Image Ratio")

        # --- CONTOUR TAB ---
        self.contour_tab = QWidget()
        self.contour_tab.setLayout(QVBoxLayout())
        self.tabs.addTab(self.contour_tab, "Contour")

        contour_group = QGroupBox("Generate Contour Masks")
        contour_layout = QGridLayout()
        contour_group.setLayout(contour_layout)

        contour_layout.addWidget(QLabel("Contour Width (pixels):"), 0, 0)
        self.contour_width_spin = QSpinBox()
        self.contour_width_spin.setRange(1, 100)
        self.contour_width_spin.setValue(5)
        contour_layout.addWidget(self.contour_width_spin, 0, 1)

        self.make_contour_btn = QPushButton("Make Contour Masks")
        self.make_contour_btn.clicked.connect(self.make_contour_masks)
        contour_layout.addWidget(self.make_contour_btn, 1, 0, 1, 2)

        self.contour_tab.layout().addWidget(contour_group)
        self.contour_tab.layout().addStretch()

        # Ratio section
        ratio_group = QGroupBox("Make Parametrical Image")
        ratio_grid = QVBoxLayout()
        ratio_group.setLayout(ratio_grid)

        ratio_info = QLabel("Colors cells by the correlation slope using settings from the Scatter Plot tab.")
        ratio_info.setWordWrap(True)
        ratio_grid.addWidget(ratio_info)
        # Options
        self.ratio_on_contours_check = QCheckBox("Calculate Ratio on Contours Only")
        self.ratio_tab.layout().addWidget(self.ratio_on_contours_check)

        self.make_ratio_btn = QPushButton("Calculate Ratio Image (All Frames)")
        self.make_ratio_btn.clicked.connect(self.make_ratio_image)
        ratio_grid.addWidget(self.make_ratio_btn)
        self.ratio_tab.layout().addWidget(ratio_group)

        # Drift Correction section
        drift_group = QGroupBox("Label Drift Correction")
        drift_grid = QGridLayout()
        drift_group.setLayout(drift_grid)

        drift_info = QLabel("Correct inconsistent IDs caused by a stage shift. Step 1: Find shift. Step 2: Apply ID remapping.")
        drift_info.setWordWrap(True)
        drift_grid.addWidget(drift_info, 0, 0, 1, 4)

        drift_grid.addWidget(QLabel("Alignment Channel:"), 1, 0)
        self.drift_img_combo = QComboBox()
        drift_grid.addWidget(self.drift_img_combo, 1, 1, 1, 3)

        drift_grid.addWidget(QLabel("Napari Index of shifted frame:"), 2, 0)
        self.drift_frame_spin = QSpinBox()
        self.drift_frame_spin.setRange(1, 10000)
        self.drift_frame_spin.setValue(89)
        drift_grid.addWidget(self.drift_frame_spin, 2, 1, 1, 3)

        self.detect_drift_btn = QPushButton("Auto-Detect Shift (Vis Only)")
        self.detect_drift_btn.clicked.connect(self.auto_detect_drift)
        drift_grid.addWidget(self.detect_drift_btn, 3, 0, 1, 2)

        self.apply_auto_drift_btn = QPushButton("Apply Auto-Detected Shift")
        self.apply_auto_drift_btn.clicked.connect(self.apply_auto_drift)
        self.apply_auto_drift_btn.setEnabled(False)
        drift_grid.addWidget(self.apply_auto_drift_btn, 3, 2, 1, 2)

        drift_grid.addWidget(QLabel("Reference ID (Frame N-1):"), 4, 0)
        self.ref_id_spin = QSpinBox()
        self.ref_id_spin.setRange(1, 100000)
        drift_grid.addWidget(self.ref_id_spin, 4, 1)

        drift_grid.addWidget(QLabel("Target ID (Frame N):"), 4, 2)
        self.target_id_spin = QSpinBox()
        self.target_id_spin.setRange(1, 100000)
        drift_grid.addWidget(self.target_id_spin, 4, 3)

        self.correct_drift_btn = QPushButton("Calculate Shift & Reassign IDs")
        self.correct_drift_btn.clicked.connect(self.apply_id_reassignment)
        drift_grid.addWidget(self.correct_drift_btn, 5, 0, 1, 4)

        self.drift_canvas = MplCanvas(self, width=3, height=3, dpi=70)
        drift_grid.addWidget(self.drift_canvas, 6, 0, 1, 4)

        self.ratio_tab.layout().addWidget(drift_group)
        self.ratio_tab.layout().addStretch()

        self.refresh_layers()

        # Connect pick event
        self.widget_canvas.mpl_connect('pick_event', self.on_pick)

        # Events
        self.viewer.layers.events.inserted.connect(self.refresh_layers)
        self.viewer.layers.events.removed.connect(self.refresh_layers)
        self.viewer.dims.events.current_step.connect(self.update_time_line)
        self.viewer.dims.events.current_step.connect(self.update_scatter_plot)
        self.tabs.currentChanged.connect(self.on_tab_change)

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
        """Timer-based debouncing for refresh_layers to avoid crashes during cleanup."""
        self._refresh_timer.start()

    def _do_refresh_layers(self):
        """The actual work of update the layer selection combos."""
        if not hasattr(self, 'image_layer_combo'):
            return

        # Block signals to prevent redundant updates during refill
        combos = [self.image_layer_combo, self.labels_layer_combo,
                  self.image_x_combo, self.image_y_combo, self.drift_img_combo]
        for c in combos:
            c.blockSignals(True)

        # Store current selections
        sel_img = self.image_layer_combo.currentText()
        sel_lbl = self.labels_layer_combo.currentText()
        sel_x = self.image_x_combo.currentText()
        sel_y = self.image_y_combo.currentText()
        sel_drift = self.drift_img_combo.currentText()

        # Clear and Refill
        for c in combos:
            c.clear()

        img_names = []
        lbl_names = []
        for layer in self.viewer.layers:
            name = layer.name
            if isinstance(layer, napari.layers.Image):
                img_names.append(name)
            elif isinstance(layer, napari.layers.Labels):
                lbl_names.append(name)

        for name in img_names:
            self.image_layer_combo.addItem(name)
            self.image_x_combo.addItem(name)
            self.image_y_combo.addItem(name)
            self.drift_img_combo.addItem(name)
        for name in lbl_names:
            self.labels_layer_combo.addItem(name)

        # Restore or set defaults
        # Image combo
        if sel_img and self.image_layer_combo.findText(sel_img) != -1:
            self.image_layer_combo.setCurrentText(sel_img)

        # Labels combo
        if sel_lbl and self.labels_layer_combo.findText(sel_lbl) != -1:
            self.labels_layer_combo.setCurrentText(sel_lbl)

        # Scatter X/Y: Restore if possible, else use smart defaults
        x_restored = False
        if sel_x and self.image_x_combo.findText(sel_x) != -1:
            self.image_x_combo.setCurrentText(sel_x)
            x_restored = True
        else:
            # Set smart default for X (405/DAPI)
            x_idx = 0
            for i, name in enumerate(img_names):
                if any(k in name.upper() for k in ["405", "DAPI"]):
                    x_idx = i
                    break
            if img_names:
                self.image_x_combo.setCurrentIndex(x_idx)
                x_restored = True

        if sel_y and self.image_y_combo.findText(sel_y) != -1:
            self.image_y_combo.setCurrentText(sel_y)
        else:
            # Set smart default for Y (488/491/GFP)
            x_name = self.image_x_combo.currentText()
            y_idx = 0 if len(img_names) < 2 or x_name != img_names[0] else 1
            for i, name in enumerate(img_names):
                if name == x_name: continue
                if any(k in name.upper() for k in ["488", "491", "GFP"]):
                    y_idx = i
                    break
            if img_names:
                self.image_y_combo.setCurrentIndex(y_idx)

        # Drift Image
        if sel_drift and self.drift_img_combo.findText(sel_drift) != -1:
            self.drift_img_combo.setCurrentText(sel_drift)
        else:
            # Prefer 491 as suggested by user
            idx = 0
            for i, name in enumerate(img_names):
                if "491" in name:
                    idx = i
                    break
            if img_names:
                self.drift_img_combo.setCurrentIndex(idx)

        # Unblock signals
        for c in combos:
            c.blockSignals(False)

        # Safety: check if current labels layer still exists
        lbl_current = self.labels_layer_combo.currentText()
        if not lbl_current or lbl_current not in self.viewer.layers:
            self.selected_labels.clear()

        # Trigger updates
        if hasattr(self, 'tabs'):
            self.update_plot()
            self.update_scatter_plot()

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

        # Safety Check: ensure layer is still in the viewer
        if lbl_name not in self.viewer.layers:
            return

        lbl_layer = self.viewer.layers[lbl_name]

        # Access data safely
        data = lbl_layer.data
        if isinstance(data, list): data = data[0]

        current_time = self.viewer.dims.current_step[0]

        # Bounds check
        if data.ndim == 3 and current_time >= data.shape[0]:
            return

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
                self.centroids_layer.border_color = 'white'
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



    def export_scatter_csv(self):
        img_x_name = self.image_x_combo.currentText()
        img_y_name = self.image_y_combo.currentText()
        lbl_name = self.labels_layer_combo.currentText()

        if not img_x_name or not img_y_name or not lbl_name:
            return

        # Safety Check: ensure layers are still in the viewer
        if img_x_name not in self.viewer.layers or \
           img_y_name not in self.viewer.layers or \
           lbl_name not in self.viewer.layers:
            return

        img_x_layer = self.viewer.layers[img_x_name]
        img_y_layer = self.viewer.layers[img_y_name]
        lbl_layer = self.viewer.layers[lbl_name]

        img_x = img_x_layer.data
        img_y = img_y_layer.data
        lbl = lbl_layer.data

        # Handle multiscale
        if isinstance(img_x, list): img_x = img_x[0]
        if isinstance(img_y, list): img_y = img_y[0]
        if isinstance(lbl, list): lbl = lbl[0]

        current_time = self.viewer.dims.current_step[0]

        # Bounds check for the time step
        if lbl.ndim == 3 and current_time >= lbl.shape[0]:
            return

        # Extract data
        if lbl.ndim == 3:
            mask_all = np.ones_like(lbl[current_time], dtype=bool)
            x_vals = img_x[current_time][mask_all]
            y_vals = img_y[current_time][mask_all]
            l_vals = lbl[current_time][mask_all]
        else:
            mask_all = np.ones_like(lbl, dtype=bool)
            x_vals = img_x[mask_all]
            y_vals = img_y[mask_all]
            l_vals = lbl[mask_all]

        # Create DataFrame
        df = pd.DataFrame({
            'X': x_vals,
            'Y': y_vals,
            'Label': l_vals
        })

        # Ask for location
        filename, _ = QFileDialog.getSaveFileName(self, "Save Scatter Data", f"scatter_{img_x_name}_{img_y_name}_frame{current_time}.csv", "CSV Files (*.csv)")
        if filename:
            df.to_csv(filename, index=False)
            print(f"Exported scatter data to {filename}")


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
        self.update_scatter_plot()

    def on_tab_change(self, index):
        if index == 0:
            self.update_plot()
        elif index == 1:
            self.update_scatter_plot()

    def make_contour_masks(self):
        lbl_name = self.labels_layer_combo.currentText()
        if not lbl_name: 
            print("Select a label layer first.")
            return
        
        lbl_layer = self.viewer.layers[lbl_name]
        data = lbl_layer.data
        if isinstance(data, list): data = data[0]
        
        width = self.contour_width_spin.value()
        
        # Hard copy to avoid modifying original
        contour_data = np.zeros_like(data)
        
        print(f"Generating contours (width={width})...")
        
        if data.ndim == 3: # T, Y, X
            for t in range(data.shape[0]):
                frame = data[t]
                if not np.any(frame): continue
                unique_ids = np.unique(frame)
                for rid in unique_ids:
                    if rid == 0: continue
                    mask = (frame == rid)
                    eroded = binary_erosion(mask, iterations=width)
                    rim = mask ^ eroded
                    contour_data[t][rim] = rid
        else: # 2D
            unique_ids = np.unique(data)
            for rid in unique_ids:
                if rid == 0: continue
                mask = (data == rid)
                eroded = binary_erosion(mask, iterations=width)
                rim = mask ^ eroded
                contour_data[rim] = rid
                
        new_name = f"{lbl_name}_contours_w{width}"
        self.viewer.add_labels(contour_data, name=new_name)
        print(f"Contour masks created: {new_name}")

    def make_ratio_image(self):
        img_x_name = self.image_x_combo.currentText()
        img_y_name = self.image_y_combo.currentText()
        lbl_name = self.labels_layer_combo.currentText()

        if not img_x_name or not img_y_name or not lbl_name:
            print("Missing layers for ratio calculation")
            return

        img_x_layer = self.viewer.layers[img_x_name]
        img_y_layer = self.viewer.layers[img_y_name]
        lbl_layer = self.viewer.layers[lbl_name]

        # Use raw data to avoid view issues
        img_x_data = img_x_layer.data
        img_y_data = img_y_layer.data
        lbl_data = lbl_layer.data
        if isinstance(img_x_data, list): img_x_data = img_x_data[0]
        if isinstance(img_y_data, list): img_y_data = img_y_data[0]
        if isinstance(lbl_data, list): lbl_data = lbl_data[0]

        # Options
        p_val = self.fit_percentile_spin.value()
        on_contours = self.ratio_on_contours_check.isChecked()
        c_width = self.contour_width_spin.value()

        ratio_data = np.zeros_like(img_x_data, dtype=float)

        print(f"Calculating Ratio Image {img_y_name}/{img_x_name} (Contours={on_contours}, Width={c_width})...")

        if img_x_data.ndim == 3: # T, Y, X
            for t in range(img_x_data.shape[0]):
                unique_labels = np.unique(lbl_data[t])
                unique_labels = unique_labels[unique_labels != 0]
                for label_id in unique_labels:
                    mask = lbl_data[t] == label_id
                    
                    # Determine which pixels to use for correlation/mean calculation
                    if on_contours:
                        # Rims only for the calculation
                        calc_mask = mask ^ binary_erosion(mask, iterations=c_width)
                    else:
                        # Specific percentile fit fallback or just mean?
                        # User says "correlation slope using settings from Scatter Plot"
                        # but in the new request for contours they say "calculate the ratio based on the contours"
                        calc_mask = mask
                    
                    val_x = img_x_data[t][calc_mask]
                    val_y = img_y_data[t][calc_mask]

                    if len(val_x) < 2: continue

                    # Use settings from scatter tab for both modes
                    cutoff_x = np.percentile(val_x, p_val)
                    cutoff_y = np.percentile(val_y, p_val)
                    fit_mask_sub = (val_x > cutoff_x) & (val_y > cutoff_y)
                    
                    x_fit = val_x[fit_mask_sub]
                    y_fit = val_y[fit_mask_sub]

                    if len(x_fit) > 2:
                        slope, _ = robust_linear_fit(x_fit, y_fit, seed=42)
                    else:
                        slope = 0
                    
                    # Fill the FULL mask with the calculated slope
                    ratio_data[t][mask] = slope
        else: # 2D
            unique_labels = np.unique(lbl_data)
            unique_labels = unique_labels[unique_labels != 0]
            for label_id in unique_labels:
                mask = lbl_data == label_id
                
                if on_contours:
                    calc_mask = mask ^ binary_erosion(mask, iterations=c_width)
                else:
                    calc_mask = mask
                
                val_x = img_x_data[calc_mask]
                val_y = img_y_data[calc_mask]

                if len(val_x) < 2: continue

                # Use settings from scatter tab
                cutoff_x = np.percentile(val_x, p_val)
                cutoff_y = np.percentile(val_y, p_val)
                fit_mask_sub = (val_x > cutoff_x) & (val_y > cutoff_y)
                x_fit = val_x[fit_mask_sub]
                y_fit = val_y[fit_mask_sub]
                
                if len(x_fit) > 2:
                    slope, _ = robust_linear_fit(x_fit, y_fit, seed=42)
                else:
                    slope = 0
                
                ratio_data[mask] = slope

        layer_name = f"Ratio {img_y_name}/{img_x_name}"
        if on_contours:
            layer_name += f" (Contour w{c_width})"
        
        self.viewer.add_image(
            ratio_data,
            name=layer_name,
            colormap='inferno',
            blending='translucent'
        )
        print(f"Ratio image added: {layer_name}")

    def auto_detect_drift(self):
        img_name = self.drift_img_combo.currentText()
        # Treat spinner value as the 0-indexed target (Napari index)
        target_idx = self.drift_frame_spin.value()
        ref_idx = target_idx - 1

        if not img_name: return
        img_data = self.viewer.layers[img_name].data
        if isinstance(img_data, list): img_data = img_data[0]

        if target_idx <= 0 or target_idx >= img_data.shape[0]:
            print(f"Invalid frame index {target_idx}")
            return

        print(f"Window correlation at Index {ref_idx} -> {target_idx}...")

        # 1. Take Reference and Current
        f_prev_raw = np.asarray(img_data[ref_idx], dtype=float)
        f_curr_raw = np.asarray(img_data[target_idx], dtype=float)

        # 2. Gaussian Blur (sigma=2)
        f_prev = gaussian_filter(f_prev_raw, sigma=2)
        f_curr = gaussian_filter(f_curr_raw, sigma=2)

        # 3. Central Window of frame 88 (50% size)
        h, w = f_prev.shape
        wh, ww = h // 2, w // 2
        y0, x0 = h // 4, w // 4
        window = f_prev[y0:y0+wh, x0:x0+ww]

        # Mean subtract to improve correlation peaks
        window_m = window - np.mean(window)
        f_curr_m = f_curr - np.mean(f_curr)

        # 4. Spatial Cross-Correlation
        # mode='valid' means window stays fully inside signal
        # correlation shape will be (h - wh + 1, w - ww + 1)
        # Using fft method for speed
        corr = correlate(f_curr_m, window_m, mode='valid', method='fft')

        # Find peak
        peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)

        # 5. Calculate Shift
        # If no shift, peak would be exactly at the 'start' of where window
        # aligns with its original position in frame 88.
        # Original window position starts at (y0, x0).
        shift_y = peak_y - y0
        shift_x = peak_x - x0

        print(f"Shift detected: y={shift_y}, x={shift_x}")
        self.last_auto_shift = (shift_y, shift_x)
        self.apply_auto_drift_btn.setEnabled(True)
        
        # --- Autofill Manual IDs for Verification ---
        lbl_name = self.labels_layer_combo.currentText()
        if lbl_name in self.viewer.layers:
            lbl_data = self.viewer.layers[lbl_name].data
            if isinstance(lbl_data, list): lbl_data = lbl_data[0]
            
            # Check if 3D
            if lbl_data.ndim == 3:
                f_ref_lbl = np.asarray(lbl_data[ref_idx])
                f_target_lbl = np.asarray(lbl_data[target_idx])
                
                # Find a label that has a match at the projected location
                unique_ids = np.unique(f_ref_lbl)
                found_sample = False
                for rid in unique_ids:
                    if rid == 0: continue
                    coords = np.argwhere(f_ref_lbl == rid)
                    ry, rx = np.mean(coords, axis=0)
                    py, px = int(round(ry + shift_y)), int(round(rx + shift_x))
                    
                    if 0 <= py < f_target_lbl.shape[0] and 0 <= px < f_target_lbl.shape[1]:
                        nid = f_target_lbl[py, px]
                        if nid != 0:
                            self.ref_id_spin.setValue(int(rid))
                            self.target_id_spin.setValue(int(nid))
                            print(f"Auto-suggested IDs for manual confirmation: Ref={rid}, Target={nid}")
                            found_sample = True
                            break
                if not found_sample:
                    print("Could not find a matching label ID for the calculated shift.")

        # Suggestion only, application now uses ID matching
        print(f"To apply manually: Note the IDs of a cell in frames {ref_idx} and {target_idx}, enter them below.")
        print("Or click 'Apply Auto-Detected Shift' to use the shift found by correlation.")

        # 6. Visualize Correlation Map
        self.drift_canvas.axes.clear()
        im = self.drift_canvas.axes.imshow(corr, cmap='viridis')
        self.drift_canvas.axes.set_title(f"Correlation Map (Max at: {peak_y}, {peak_x})")
        # Add a dot at peak
        self.drift_canvas.axes.plot(peak_x, peak_y, 'ro')
        self.drift_canvas.draw()

    def apply_auto_drift(self):
        # Triggered by the new button
        target_idx = self.drift_frame_spin.value()
        sy, sx = self.last_auto_shift
        self._perform_label_reassignment(sy, sx, target_idx)

    def apply_id_reassignment(self):
        lbl_name = self.labels_layer_combo.currentText()
        # Treat the spinner value as the 0-indexed target frame (Napari index)
        target_idx = self.drift_frame_spin.value()
        ref_idx = target_idx - 1

        rid_ref = self.ref_id_spin.value()
        rid_target = self.target_id_spin.value()

        if not lbl_name: return
        lbl_layer = self.viewer.layers[lbl_name]

        # Handle multiscale and cast to numpy array (hard copy to be safe)
        lbl_data_source = lbl_layer.data
        if isinstance(lbl_data_source, list):
            lbl_data_source = lbl_data_source[0]
        
        # Check bounds
        if target_idx <= 0 or target_idx >= lbl_data_source.shape[0]:
            print(f"Index {target_idx} is out of bounds for the stack.")
            return

        # 1. Calculate Shift from centroids of the two IDs
        f_ref = np.asarray(lbl_data_source[ref_idx])
        f_target = np.asarray(lbl_data_source[target_idx])

        # Robust existence check using np.any
        if not np.any(f_ref == rid_ref):
            print(f"Error: Label {rid_ref} not found in Reference frame (Index {ref_idx})")
            return
        if not np.any(f_target == rid_target):
            print(f"Error: Label {rid_target} not found in Target frame (Index {target_idx})")
            return

        coords_ref = np.argwhere(f_ref == rid_ref)
        coords_target = np.argwhere(f_target == rid_target)

        cy_ref, cx_ref = np.mean(coords_ref, axis=0)
        cy_target, cx_target = np.mean(coords_target, axis=0)

        shift_y = cy_target - cy_ref
        shift_x = cx_target - cx_ref

        print(f"Mapping built from Manual IDs at Index {ref_idx} -> {target_idx}")
        print(f"Found centroids: Ref({cy_ref:.1f},{cx_ref:.1f}) | Target({cy_target:.1f},{cx_target:.1f})")
        print(f"Shift for re-linking: dy={shift_y:.2f}, dx={shift_x:.2f}")

        self._perform_label_reassignment(shift_y, shift_x, target_idx)

    def _perform_label_reassignment(self, shift_y, shift_x, target_idx):
        lbl_name = self.labels_layer_combo.currentText()
        if not lbl_name: return
        lbl_layer = self.viewer.layers[lbl_name]
        
        lbl_data_source = lbl_layer.data
        if isinstance(lbl_data_source, list):
            lbl_data_source = lbl_data_source[0]
            
        # Hard copy to modify
        lbl_data = np.asarray(lbl_data_source).copy()
        ref_idx = target_idx - 1
        f_ref = lbl_data[ref_idx]
        f_target = lbl_data[target_idx]

        # 2. Build mapping ONCE
        mapping = {}
        unique_ref = np.unique(f_ref)
        for rid in unique_ref:
            if rid == 0: continue
            coords = np.argwhere(f_ref == rid)
            ry, rx = np.mean(coords, axis=0)

            # Project reference centroid exactly as it should look in target frame
            py = int(round(ry + shift_y))
            px = int(round(rx + shift_x))

            if 0 <= py < f_target.shape[0] and 0 <= px < f_target.shape[1]:
                nid = f_target[py, px]
                if nid != 0:
                    mapping[nid] = rid

        print(f"Building re-mapping for {len(mapping)} labels using shift ({shift_y:.2f}, {shift_x:.2f})...")

        # 3. Apply ONLY to frames from target_idx onwards
        for t in range(target_idx, lbl_data.shape[0]):
            curr_f = lbl_data[t]
            # Use a mask based approach to avoid multiple overwrites
            remap = curr_f.copy()
            for nid, rid in mapping.items():
                remap[curr_f == nid] = rid
            lbl_data[t] = remap

        lbl_layer.data = lbl_data
        lbl_layer.refresh()
        print(f"Label ID correction complete. Frames {target_idx} to {lbl_data.shape[0]-1} were updated.")

    def update_scatter_plot(self, event=None):
        if self.tabs.currentIndex() != 1:
            return

        self.scatter_canvas.axes.clear()
        self.scatter_canvas.axes.patch.set_alpha(0)

        img_x_name = self.image_x_combo.currentText()
        img_y_name = self.image_y_combo.currentText()
        lbl_name = self.labels_layer_combo.currentText()

        if not img_x_name or not img_y_name or not lbl_name:
            return

        # Safety Check: ensure layers are still in the viewer
        if img_x_name not in self.viewer.layers or \
           img_y_name not in self.viewer.layers or \
           lbl_name not in self.viewer.layers:
            return

        img_x_layer = self.viewer.layers[img_x_name]
        img_y_layer = self.viewer.layers[img_y_name]
        lbl_layer = self.viewer.layers[lbl_name]

        img_x = img_x_layer.data
        img_y = img_y_layer.data
        lbl = lbl_layer.data

        # Handle multiscale
        if isinstance(img_x, list): img_x = img_x[0]
        if isinstance(img_y, list): img_y = img_y[0]
        if isinstance(lbl, list): lbl = lbl[0]

        current_time = self.viewer.dims.current_step[0]

        # Bounds check for the time step
        if lbl.ndim == 3 and current_time >= lbl.shape[0]:
            return
        alpha = self.alpha_spin.value()
        psize = self.psize_spin.value()

        # Use a fixed random state for background subsampling to ensure stability
        rng = np.random.RandomState(42)

        # --- Plot Background if requested ---
        if self.show_bg_check.isChecked():
            if lbl.ndim == 3:
                bg_mask = lbl[current_time] == 0
                bg_x = img_x[current_time][bg_mask]
                bg_y = img_y[current_time][bg_mask]
            else:
                bg_mask = lbl == 0
                bg_x = img_x[bg_mask]
                bg_y = img_y[bg_mask]

            # Subsample for plotting performance (using fixed rng)
            if len(bg_x) > 5000:
                indices_plot = rng.choice(len(bg_x), 5000, replace=False)
                bg_x_plot = bg_x[indices_plot]
                bg_y_plot = bg_y[indices_plot]
            else:
                bg_x_plot, bg_y_plot = bg_x, bg_y

            self.scatter_canvas.axes.scatter(bg_x_plot, bg_y_plot, alpha=alpha, s=psize, color='red', edgecolors='none', label='Background')

            # --- Background Fit (Robust Deterministic) ---
            if self.fit_bg_check.isChecked() and len(bg_x) > 10:
                # We use robust_linear_fit (Soft L1 loss) to ignore outliers
                # that would pull a simple OLS line to a negative slope.
                # We take a large enough subsample (10k) for high precision and speed.
                if len(bg_x) > 10000:
                    indices_fit = rng.choice(len(bg_x), 10000, replace=False)
                    bx_fit, by_fit = bg_x[indices_fit], bg_y[indices_fit]
                else:
                    bx_fit, by_fit = bg_x, bg_y

                # Deterministic robust fit
                a_bg, b_bg = robust_linear_fit(bx_fit, by_fit, seed=42)

                # Plot line over the visible range of background data
                x_min, x_max = np.min(bg_x), np.max(bg_x)
                x_range = np.array([x_min, x_max])
                y_range = a_bg * x_range + b_bg
                self.scatter_canvas.axes.plot(x_range, y_range, color='red', linestyle='--', linewidth=1.5, label='BG Fit')

                bg_eq = f"BG: y = {a_bg:.3f}x + {b_bg:.1f} (Robust)"
                self.scatter_canvas.axes.text(0.05, 0.85, bg_eq, color='red', transform=self.scatter_canvas.axes.transAxes,
                                              verticalalignment='top', fontweight='bold', fontsize=10)

        if not self.selected_labels:
            if not self.show_bg_check.isChecked():
                self.scatter_canvas.axes.text(0.5, 0.5, "Select a label to see scatter plot",
                                              color='white', ha='center', va='center',
                                              transform=self.scatter_canvas.axes.transAxes)
            self.scatter_canvas.draw()
            return

        # Get the "last" selected label for scatter
        label_id = list(self.selected_labels)[-1]
        
        on_contours = self.scatter_on_contours_check.isChecked()
        c_width = self.contour_width_spin.value()

        # Check if 3D (T, Y, X)
        if lbl.ndim == 3:
            mask = lbl[current_time] == label_id
            if np.any(mask):
                if on_contours:
                    # Rims only
                    mask = mask ^ binary_erosion(mask, iterations=c_width)
                
                val_x = img_x[current_time][mask]
                val_y = img_y[current_time][mask]

                # --- Logical Split for Fitting ---
                if self.perform_fit_check.isChecked():
                    p_val = self.fit_percentile_spin.value()
                    cutoff_x = np.percentile(val_x, p_val)
                    cutoff_y = np.percentile(val_y, p_val)

                    fit_mask = (val_x > cutoff_x) & (val_y > cutoff_y)
                    bg_points_mask = ~fit_mask

                    # Plot points BELOW cutoff (Cyan)
                    self.scatter_canvas.axes.scatter(val_x[bg_points_mask], val_y[bg_points_mask],
                                                     alpha=alpha, s=psize, color='cyan', edgecolors='none')

                    # Plot points ABOVE cutoff (Magenta) - these are being fitted
                    self.scatter_canvas.axes.scatter(val_x[fit_mask], val_y[fit_mask],
                                                     alpha=alpha, s=psize, color='magenta', edgecolors='none', label='Fitted Points')

                    x_fit = val_x[fit_mask]
                    y_fit = val_y[fit_mask]

                    if len(x_fit) > 2:
                        # Robust Linear Regression (Deterministic with seed)
                        a, b = robust_linear_fit(x_fit, y_fit, seed=42)

                        # Plot fit line
                        x_range = np.array([0, np.max(val_x)])
                        y_range = a * x_range + b
                        self.scatter_canvas.axes.plot(x_range, y_range, color='gold', linestyle='--', linewidth=2, label='Fit')

                        # Display equation
                        eq_text = f"y = {a:.3f}x + {b:.1f}"
                        self.scatter_canvas.axes.text(0.05, 0.95, eq_text, color='gold', transform=self.scatter_canvas.axes.transAxes,
                                                      verticalalignment='top', fontweight='bold', fontsize=12)
                else:
                    # Default cyan scatter for everything if fit is off
                    self.scatter_canvas.axes.scatter(val_x, val_y, alpha=alpha, s=psize, color='cyan', edgecolors='none', label=f'Label {label_id}')

                self.scatter_canvas.axes.set_title(f"Label {label_id} at Frame {current_time}", color='white')
                self.scatter_canvas.axes.set_xlabel(img_x_name, color='white')
                self.scatter_canvas.axes.set_ylabel(img_y_name, color='white')
                self.scatter_canvas.axes.set_xlim(left=0)
                self.scatter_canvas.axes.set_ylim(bottom=0)
                self.scatter_canvas.axes.tick_params(colors='white')

                for spine in self.scatter_canvas.axes.spines.values():
                    spine.set_color('white')
            else:
                self.scatter_canvas.axes.set_title(f"Label {label_id} not present at Frame {current_time}", color='white')

        elif lbl.ndim == 2:
            mask = lbl == label_id
            if np.any(mask):
                if on_contours:
                    mask = mask ^ binary_erosion(mask, iterations=c_width)
                
                val_x = img_x[mask]
                val_y = img_y[mask]

                if self.perform_fit_check.isChecked():
                    p_val = self.fit_percentile_spin.value()
                    cutoff_x = np.percentile(val_x, p_val)
                    cutoff_y = np.percentile(val_y, p_val)
                    fit_mask = (val_x > cutoff_x) & (val_y > cutoff_y)
                    bg_points_mask = ~fit_mask

                    self.scatter_canvas.axes.scatter(val_x[bg_points_mask], val_y[bg_points_mask], alpha=alpha, s=psize, color='cyan', edgecolors='none')
                    self.scatter_canvas.axes.scatter(val_x[fit_mask], val_y[fit_mask], alpha=alpha, s=psize, color='magenta', edgecolors='none', label='Fitted Points')

                    x_fit = val_x[fit_mask]
                    y_fit = val_y[fit_mask]
                    if len(x_fit) > 2:
                        # Robust Linear Regression (Deterministic with seed)
                        a, b = robust_linear_fit(x_fit, y_fit, seed=42)
                        x_range = np.array([0, np.max(val_x)])
                        y_range = a * x_range + b
                        self.scatter_canvas.axes.plot(x_range, y_range, color='gold', linestyle='--', linewidth=2, label='Fit')
                        eq_text = f"y = {a:.3f}x + {b:.1f}"
                        self.scatter_canvas.axes.text(0.05, 0.95, eq_text, color='gold', transform=self.scatter_canvas.axes.transAxes,
                                                      verticalalignment='top', fontweight='bold', fontsize=12)
                else:
                    self.scatter_canvas.axes.scatter(val_x, val_y, alpha=alpha, s=psize, color='cyan', edgecolors='none', label=f'Label {label_id}')

                self.scatter_canvas.axes.set_title(f"Label {label_id}", color='white')
                self.scatter_canvas.axes.set_xlabel(img_x_name, color='white')
                self.scatter_canvas.axes.set_ylabel(img_y_name, color='white')
                self.scatter_canvas.axes.set_xlim(left=0)
                self.scatter_canvas.axes.set_ylim(bottom=0)
                self.scatter_canvas.axes.tick_params(colors='white')
                for spine in self.scatter_canvas.axes.spines.values():
                    spine.set_color('white')

        self.scatter_canvas.draw()

    def update_plot(self):
        img_name = self.image_layer_combo.currentText()
        lbl_name = self.labels_layer_combo.currentText()

        if not img_name or not lbl_name:
            return

        # Safety Check: ensure layers are still in the viewer
        if img_name not in self.viewer.layers or lbl_name not in self.viewer.layers:
            return

        img_layer = self.viewer.layers[img_name]
        lbl_layer = self.viewer.layers[lbl_name]

        # Access data safely (handling dask/multiscale)
        img_data_raw = img_layer.data
        lbl_data_raw = lbl_layer.data
        if isinstance(img_data_raw, list): img_data_raw = img_data_raw[0]
        if isinstance(lbl_data_raw, list): lbl_data_raw = lbl_data_raw[0]

        self.widget_canvas.axes.clear()
        self.widget_canvas.axes.patch.set_alpha(0)

        # Get options
        show_markers = self.show_markers_check.isChecked()
        linewidth = self.linewidth_spin.value()
        normalize = self.normalize_check.isChecked()
        smooth = self.smooth_check.isChecked()
        smooth_window = self.smooth_spin.value()
        show_avg = self.show_avg_check.isChecked()
        show_bg_ts = self.show_bg_ts_check.isChecked()

        marker = 'o' if show_markers else None
        markersize = linewidth * 3 # Adapt marker size

        # --- Population Average Plotting ---
        if show_avg:
            # DO NOT use np.asarray() on the full 3D stack.
            # Use lazy access to frames if it's dask
            if img_data_raw.ndim == 3:
                all_means = []
                for t in range(img_data_raw.shape[0]):
                    mask = lbl_data_raw[t] > 0
                    if np.any(mask):
                        all_means.append(np.mean(img_data_raw[t][mask]))
                    else:
                        all_means.append(np.nan)

                avg_intensities = all_means
                avg_times = list(range(len(avg_intensities)))

                if smooth and len(avg_intensities) > smooth_window:
                    window = np.ones(smooth_window) / smooth_window
                    avg_intensities = np.convolve(avg_intensities, window, mode='valid')
                    offset = (smooth_window - 1) // 2
                    avg_times = avg_times[offset : offset + len(avg_intensities)]

                if normalize:
                    avg_intensities = (avg_intensities - np.nanmin(avg_intensities)) / (np.nanmax(avg_intensities) - np.nanmin(avg_intensities) + 1e-8)

                self.widget_canvas.axes.plot(avg_times, avg_intensities, color='white', linestyle='--', linewidth=linewidth*1.5, label='Population Avg')

        # --- Background Plotting ---
        if show_bg_ts:
            if img_data_raw.ndim == 3:
                bg_means = []
                for t in range(img_data_raw.shape[0]):
                    bg_mask = lbl_data_raw[t] == 0
                    if np.any(bg_mask):
                        bg_means.append(np.mean(img_data_raw[t][bg_mask]))
                    else:
                        bg_means.append(np.nan)

                bg_intensities = np.array(bg_means)
                bg_times = list(range(len(bg_intensities)))

                if smooth and len(bg_intensities) > smooth_window:
                    window = np.ones(smooth_window) / smooth_window
                    bg_intensities = np.convolve(bg_intensities, window, mode='valid')
                    offset = (smooth_window - 1) // 2
                    bg_times = bg_times[offset : offset + len(bg_intensities)]

                if normalize:
                    bg_intensities = (bg_intensities - np.nanmin(bg_intensities)) / (np.nanmax(bg_intensities) - np.nanmin(bg_intensities) + 1e-8)

                self.widget_canvas.axes.plot(bg_times, bg_intensities, color='red', linestyle=':', linewidth=linewidth*1.5, label='Background')

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




