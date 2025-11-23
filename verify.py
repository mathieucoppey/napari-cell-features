import napari
import numpy as np
from napari_cell_features import CellFeaturesWidget

def create_synthetic_data():
    # Create synthetic 2D+t data
    # T=10, Y=100, X=100
    image_data = np.random.rand(10, 100, 100)
    
    # Create moving blob
    labels_data = np.zeros((10, 100, 100), dtype=int)
    for t in range(10):
        # Blob moves diagonally
        y, x = 20 + t*2, 20 + t*2
        labels_data[t, y:y+10, x:x+10] = 1
        
        # Another blob
        y2, x2 = 80 - t*2, 20 + t
        labels_data[t, y2:y2+10, x2:x2+10] = 2

    return image_data, labels_data

def main():
    viewer = napari.Viewer()
    image, labels = create_synthetic_data()
    
    viewer.add_image(image, name='Synthetic Image')
    viewer.add_labels(labels, name='Synthetic Labels')
    
    widget = CellFeaturesWidget(viewer)
    viewer.window.add_dock_widget(widget, name='Cell Features')
    
    print("Plugin loaded. Click on the blobs to test plotting.")
    napari.run()

if __name__ == "__main__":
    main()
