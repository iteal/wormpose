"""
Simple FeaturesDataset implementation that reads features from a HDF5 file
"""

import os
from typing import List

import h5py
from glob import glob

from wormpose.dataset.base_dataset import BaseFeaturesDataset
from wormpose.pose.centerline import get_joint_indexes


class HDF5Features(BaseFeaturesDataset):
    def __init__(self, dataset_path: str, video_names: List[str]):

        features_file = glob(os.path.join(dataset_path, "*.h*5"))
        if len(features_file) == 0:
            raise FileNotFoundError(f"Can't load features, missing .h5 file in {dataset_path}")

        with h5py.File(features_file[0], "r") as f:
            width = f["width"][:]
            skeletons = f["skeletons"][:]
            framerate = f.attrs["framerate"]

        head_joint, mid_body_joint, tail_body_joint = get_joint_indexes(len(width[0]))

        features_dict = {
            "skeletons": skeletons,
            "head_width": width[:, head_joint],
            "midbody_width": width[:, mid_body_joint],
            "tail_width": width[:, tail_body_joint],
            "frame_rate": framerate,
        }

        self._features = {video_names[0]: features_dict}

    def get_features(self, video_name: str):
        return self._features[video_name]
