"""
Loaders for Tierpsy tracker data

The files should be organized this way :

+-- dataset_path
|   +-- video_name0
|   |   +-- video_name0.hdf5
|   |   +-- video_name0_features.hdf5  or video_name0_featuresN.hdf5
|   +-- video_name1
|   |   +-- video_name1.hdf5
|   |   +-- video_name1_features.hdf5 or video_name1_featuresN.hdf5
|   ...

WARNING not all configurations of Tierpsy tracker are supported.
For example: only one worm per file will be extracted (the one with the smallest index)
"""

from wormpose.dataset.image_processing.simple_frame_preprocessing import SimpleFramePreprocessing

FramePreprocessing = SimpleFramePreprocessing
from wormpose.dataset.loaders.tierpsy.frames_dataset import FramesDataset
from wormpose.dataset.loaders.tierpsy.features_dataset import FeaturesDataset
from wormpose.dataset.loaders.tierpsy.results_exporter import ResultsExporter
