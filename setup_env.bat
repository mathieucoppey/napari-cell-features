call conda create -n napari-cell-features python=3.10 -y
call conda activate napari-cell-features
call pip install napari[all] pandas matplotlib numpy pyqtgraph
call pip install -e .
