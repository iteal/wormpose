"""
Simple FramesDataset implementation reading image files from a folder

Only one video is supported (one folder of images)
"""

import glob
import os
import cv2

from wormpose.dataset.base_dataset import BaseFramesDataset


class ImagesFolder(BaseFramesDataset):
    """
    Simple loading images in memory from one folder (one video_name)
    """

    class FramesReader(object):
        def __init__(self, filenames):
            self.data = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in filenames]

        def __enter__(self):
            return self.data

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    def __init__(self, dataset_path: str, extension: str = "*.png"):
        self.filenames = sorted(glob.glob(os.path.join(dataset_path, extension)))
        if len(self.filenames) == 0:
            raise FileNotFoundError(f"Can't find {extension} images in folder {dataset_path}")

        self.video_name = os.path.basename(os.path.normpath(dataset_path))

    def video_names(self):
        return [self.video_name]

    def open(self, _):
        return self.FramesReader(self.filenames)
