"""
Dataset loader for the tutorial sample data
"""

from wormpose.dataset.image_processing.simple_frame_preprocessing import SimpleFramePreprocessing
from wormpose.dataset.loaders.hdf5features import HDF5Features
from wormpose.dataset.loaders.images_folder import ImagesFolder

FramesDataset = ImagesFolder
FeaturesDataset = HDF5Features
FramePreprocessing = SimpleFramePreprocessing
