"""
Implementation of BaseFramesDataset to load Tierpsy tracker frames
"""

import os

import h5py

from wormpose import BaseFramesDataset


class FramesDataset(BaseFramesDataset):
    class FramesReader(object):
        def __init__(self, h5_filename: str):
            self.h5_filename = h5_filename

        def __enter__(self):
            self.f = h5py.File(self.h5_filename, "r")
            return self.f["mask"]

        def __exit__(self, exc_type, exc_value, traceback):
            self.f.close()

    def __init__(self, dataset_path: str):

        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"'{dataset_path}' needs to be a directory containing Tierpsy videos.")

        self.videos_paths = {}

        for video_name in os.listdir(dataset_path):
            if os.path.isdir(os.path.join(dataset_path, video_name)):
                video_path = os.path.join(dataset_path, video_name, video_name + ".hdf5")
                if os.path.exists(video_path):
                    self.videos_paths[video_name] = video_path

        if len(self.videos_paths) == 0:
            raise FileNotFoundError(f"Couldn't find Tierpsy videos in '{dataset_path}'.")

    def video_names(self):
        return list(self.videos_paths.keys())

    def open(self, video_name: str):
        return self.FramesReader(self.videos_paths[video_name])
