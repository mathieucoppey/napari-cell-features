# napari-cell-features

A Napari plugin for extracting and visualizing features from mammalian cells (2D+t) using images and tracked masks.

## Features

- **Time Series Plotting**: Visualize mean intensity over time for selected cells.
- **Interactive Selection**: Click on cells in the viewer to select/deselect them.
- **Centroid Markers**: Automatically displays a marker at the centroid of selected cells that moves with time.
- **Multi-Selection**: Compare multiple cells simultaneously.
- **Data Export**: Export the time-series data of selected cells to a CSV file.
- **Customization**: Options for normalization, smoothing, and plot appearance.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/YOUR_USERNAME/napari-cell-features.git
    cd napari-cell-features
    ```

2.  Install the plugin:
    ```bash
    pip install -e .
    ```

## Usage

1.  Open Napari:
    ```bash
    napari
    ```
2.  Load your 2D+t Image and Labels layers.
3.  Go to **Plugins > napari-cell-features > Cell Features**.
4.  Select the correct **Image Layer** and **Labels Layer** in the widget.
5.  **Click** on a cell in the viewer to plot its intensity over time.
    - A white ring marker will appear on the cell.
    - The plot will update with the cell's data.
6.  **Select multiple cells** to compare them.
7.  Use the **Plot Options** to smooth or normalize the data.
8.  Click **Export CSV** to save the data for analysis.

## Requirements

- napari
- pandas
- matplotlib
- numpy
- qtpy
- magicgui
- scipy
