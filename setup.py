from setuptools import setup, find_packages

setup(
    name='napari-cell-features',
    version='0.0.1',
    packages=find_packages(),
    entry_points={
        'napari.manifest': [
            'napari-cell-features = napari_cell_features:napari.yaml',
        ],
    },
    package_data={'napari_cell_features': ['napari.yaml']},
    install_requires=[
        'napari',
        'pandas',
        'matplotlib',
        'numpy',
        'qtpy',
        'magicgui',
        'scipy' 
    ],
)
