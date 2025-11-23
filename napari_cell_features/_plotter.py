import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QDialog
from PyQt5.QtCore import Qt
import numpy as np

def extract_time_series(image_layer, labels_layer, label_id):
    """
    Extract mean intensity over time for a given label.
    Assumes image_layer is (T, Y, X) or (T, C, Y, X) and labels_layer is (T, Y, X).
    """
    # Simple implementation assuming T, Y, X for now
    # TODO: Handle multi-channel and other dimensions more robustly
    
    data = image_layer.data
    labels = labels_layer.data
    
    # Check dimensions
    if data.ndim == 3: # T, Y, X
        times = range(data.shape[0])
        intensities = []
        for t in times:
            mask = labels[t] == label_id
            if np.any(mask):
                mean_val = np.mean(data[t][mask])
                intensities.append(mean_val)
            else:
                intensities.append(np.nan)

        return times, intensities
    elif data.ndim == 4: # T, C, Y, X or C, T, Y, X? Usually napari is T, Y, X or T, Z, Y, X
        # Assuming T, Z, Y, X or similar, need to be careful. 
        # For 2D+t, it's usually (T, Y, X). 
        # If 4D, maybe (T, Z, Y, X).
        # Let's stick to simple 3D (T, Y, X) for the first pass as requested (2D+t).
        pass
    
    return [], []

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_alpha(0) # Transparent figure background
        self.axes = fig.add_subplot(111)
        self.axes.patch.set_alpha(0) # Transparent axes background
        super(MplCanvas, self).__init__(fig)
        self.setStyleSheet("background-color:transparent;")

class PopupPlot(QDialog):
    def __init__(self, parent=None, title="Cell Features"):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        layout = QVBoxLayout()
        self.canvas = MplCanvas(self, width=4, height=3, dpi=100)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.setWindowTitle(title)

    def plot(self, x, y, label_id):
        self.canvas.axes.clear()
        self.canvas.axes.plot(x, y, marker='o')
        self.canvas.axes.set_title(f"Label {label_id} Intensity over Time")
        self.canvas.axes.set_xlabel("Time")
        self.canvas.axes.set_ylabel("Mean Intensity")
        self.canvas.draw()
