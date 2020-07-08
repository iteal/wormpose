"""
This module deals with loading features from a dataset.

It will calculate extra features such as the worm length.
"""

from typing import Dict, Tuple

import numpy as np


class Features(object):
    def __init__(self, raw_features: dict):

        _validate_features(raw_features)

        self.skeletons = raw_features["skeletons"]
        self.frame_rate = raw_features["frame_rate"]
        self.timestamp = raw_features["timestamp"] if "timestamp" in raw_features else None
        self.ventral_side = raw_features["ventral_side"] if "ventral_side" in raw_features else None

        # set defaults for optional values
        if self.ventral_side is None:
            self.ventral_side = "unknown"
        if self.timestamp is None:  # default behavior: all frames are equidistant in time
            self.timestamp = np.arange(0, len(self.skeletons))

        self._set_measurements(raw_features)
        self._set_labelled_indexes()

    def _set_labelled_indexes(self):
        # Save indexes where skeleton is valid (not nan)
        skel_is_not_nan = ~np.any(np.isnan(self.skeletons), axis=(1, 2))
        self.labelled_indexes = np.where(skel_is_not_nan)[0]

    def _set_measurements(self, raw_features: dict):
        worm_length = _calculate_worm_length(self.skeletons)
        self.measurements = np.stack(
            [worm_length, raw_features["head_width"], raw_features["midbody_width"], raw_features["tail_width"],],
            axis=1,
        )
        self.measurements.dtype = {
            "names": ("worm_length", "head_width", "midbody_width", "tail_width"),
            "formats": (float, float, float, float),
        }


FeaturesDict = Dict[str, Features]


def _calculate_worm_length(skeletons):
    worm_length = np.full(len(skeletons), np.nan, dtype=float)
    where_skel_nan = np.where(~np.any(np.isnan(skeletons), axis=(1, 2)))
    for i in where_skel_nan[0]:
        skel = skeletons[i]
        worm_length[i] = np.sum(np.sqrt(np.sum((skel[:-1] - skel[1:]) ** 2, axis=1)))
    return worm_length


def _validate_features(raw_features: dict):
    if not (
        len(raw_features["skeletons"])
        == len(raw_features["head_width"])
        == len(raw_features["midbody_width"])
        == len(raw_features["tail_width"])
    ):
        raise ValueError("inconsistent features")
    if not (
        raw_features["head_width"].dtype == raw_features["midbody_width"].dtype == raw_features["tail_width"].dtype
    ) or not np.issubdtype(raw_features["head_width"].dtype, np.floating):
        raise TypeError("Body measurements type should be identical and floating point")
    if not np.issubdtype(raw_features["skeletons"].dtype, np.floating):
        raise TypeError("Skeleton type should be floating point")
    if len(raw_features["skeletons"].shape) != 3 or raw_features["skeletons"].shape[2] != 2:
        raise ValueError("Wrong skeleton shape")
    if raw_features["skeletons"].shape[1] < 20:
        raise UserWarning(
            "Low number of skeleton joints (< 20), consider interpolating to improve quality of synthetic images"
        )
    if raw_features.get("timestamp") is not None and len(raw_features["timestamp"]) != len(raw_features["skeletons"]):
        raise ValueError("Inconsistent timestamp")


def calculate_max_average_worm_length(features: FeaturesDict) -> float:
    """
    Calculates the average worm length from each video, and returns the maximum.
    
    :param features: A dictionary of Features
    :return: Biggest average worm length from all videos
    """
    return float(np.nanmax([np.nanmean(x.measurements["worm_length"]) for x in features.values()]))


MINIMUM_IMAGE_SIZE = 32


def calculate_crop_window_size(features: FeaturesDict) -> Tuple[int, int]:
    """
    Returns an image shape that is just big enough to view the worm object,
    the image will be cropped (or expanded) to that size as an input to the neural network
    Can be overriden in child classes for another behavior (for example a fixed chosen size)
    :param features: A dictionary of Features
    :return: A tuple of two integer values (height, width)
    """
    import math

    # calculate the image shape as maximum of all the average worm lengths for each video
    # then we know the longest worm will fit the image shape,
    # the smaller ones will just have more background space
    max_average_worm_length = calculate_max_average_worm_length(features)
    if np.isnan(max_average_worm_length):
        raise ValueError(
            "Can't calculate the crop window size: "
            "couldn't get the max average worm length in the dataset."
            " Please check the labeled features in this dataset."
        )

    crop_size = max(MINIMUM_IMAGE_SIZE, int(max_average_worm_length))

    # round to even number
    crop_size = math.ceil(float(crop_size) / 2) * 2
    return crop_size, crop_size
