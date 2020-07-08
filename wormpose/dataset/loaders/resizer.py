"""
Handles all about the optional resizing of the images of a Dataset

It will modify the FeaturesDataset and the FramesDataset to update with the resized values.

The resize parameter can be specified as a single number (ex: 2 for upscale to twice the size, or 0.5 to downscale half the size)
or as the desired image size, in that case, the resize factor will be calculated depending on the average worm length
in the dataset.
"""

import copy
from typing import Type

import cv2
from wormpose.dataset.base_dataset import BaseFeaturesDataset, BaseFramesDataset

from wormpose.dataset.features import (
    calculate_max_average_worm_length,
    calculate_crop_window_size,
    MINIMUM_IMAGE_SIZE,
)


class ResizeOptions(object):
    """
    Class handling the possible resize options: either a scaling factor, or a set image size
    """

    def __init__(self, resize_factor: float = None, image_size: int = None, **kwargs):
        self.resize_factor = resize_factor
        self._image_size = image_size

        # validate
        if self._image_size is not None:
            if self._image_size % 2 != 0:
                raise NotImplementedError("If image size is set, it should be an even number")
            if self._image_size < MINIMUM_IMAGE_SIZE:
                raise ValueError(f"Image size must be bigger than {MINIMUM_IMAGE_SIZE}")

    def update_resize_factor(self, features_dataset):

        if self.resize_factor is not None:
            return

        if self._image_size is not None:
            worm_length = calculate_max_average_worm_length(features_dataset)
            self.resize_factor = self._image_size / worm_length
            return

        self.resize_factor = 1.0

    def get_image_shape(self, features_dataset):
        if self._image_size is not None:
            return self._image_size, self._image_size

        return calculate_crop_window_size(features_dataset)


def add_resizing_arguments(parser):
    """
    For command line arguments, adds two mutually exclusive resize options: either a scaling factor, or a set image size

    :param parser: Argparse parser
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resize_factor",
        type=float,
        help="Option to resize the images. "
        "Examples: '0.5' downscale half, '2' upscale twice, '1' no resizing (default)",
    )
    group.add_argument(
        "--image_size",
        type=int,
        help="Option to resize the images so that the average worm of the dataset "
        "fits into a square image of side: image_size",
    )


class ResizedFrames(object):
    def __init__(self, scale_factor: float, frames):
        self.frames = frames
        self.scale_factor = scale_factor

    def _resize(self, frame):
        return cv2.resize(frame, dsize=None, fx=self.scale_factor, fy=self.scale_factor)

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError("Slicing notation not implemented with resizer.")
        orig = self.frames[key]
        return self._resize(orig)

    def __len__(self):
        return len(self.frames)


class ResizedFramesReader(object):
    def __init__(self, frames_reader, resize_factor: float):
        self.frames_reader = frames_reader
        self.resize_factor = resize_factor

    def __enter__(self):
        self.resized_frames = ResizedFrames(self.resize_factor, self.frames_reader.__enter__())
        return self.resized_frames

    def __exit__(self, exc_type, exc_value, traceback):
        self.resized_frames = None
        self.frames_reader.__exit__(exc_type, exc_value, traceback)


def frames_dataset_resizer(frames_dataset_class: Type[BaseFramesDataset], resize_factor: float):
    """
    Modifies a FramesDataset class so that all frames are resized by resize_factor

    :param frames_dataset_class: original FramesDataset class without resizing
    :param resize_factor: by how much to rescale the images
    :return: new FramesDataset with resizing
    """
    old_open = frames_dataset_class.open

    def resizer_open(*args):
        frames_reader = old_open(*args)
        return ResizedFramesReader(frames_reader, resize_factor=resize_factor)

    frames_dataset_class.open = resizer_open
    return frames_dataset_class


def features_dataset_resizer(features_dataset_class: Type[BaseFeaturesDataset], resize_factor: float):
    """
    Modifies a FeaturesDataset class so that all features are resized by resize_factor

    :param frames_dataset_class: original FeaturesDataset class without resizing
    :param resize_factor: by how much to rescale the features
    :return: new FeaturesDataset with resizing
    """
    old_get = features_dataset_class.get_features

    def resizer_get(*args):
        features = old_get(*args)
        new_features = copy.deepcopy(features)
        new_features["skeletons"] *= resize_factor
        new_features["head_width"] *= resize_factor
        new_features["midbody_width"] *= resize_factor
        new_features["tail_width"] *= resize_factor
        return new_features

    features_dataset_class.get_features = resizer_get
    return features_dataset_class
