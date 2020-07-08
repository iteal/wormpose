"""
Applies safely the frame preprocessing function to a frame,
set the background pixels to a uniform value, deduces the region of interest
"""
from typing import Tuple

import numpy as np

from wormpose.dataset.base_dataset import BaseFramePreprocessing


def run(frame_preprocessing: BaseFramePreprocessing, frame: np.ndarray) -> Tuple[np.ndarray, int, Tuple[slice, slice]]:
    """
    Safely preprocesses an image, set the background pixels to a uniform color, calculates worm region of interest

    :param frame_preprocessing: Frame preprocessing logic
    :param frame: Image to preprocess
    :return: Processed image, value of the background color, region of interest coordinates
    """

    # copy to avoid modifying the source image
    frame_copy = np.copy(frame)

    # call the frame preprocessing function to get the segmented image
    segmentation_mask, background_color = frame_preprocessing.process(frame_copy)

    # enforces background color type
    background_color = int(background_color)

    # erase background, set everything not the foreground to a uniform color
    frame_copy[segmentation_mask == 0] = background_color

    # get region of interest (full image if no worm is found)
    where_worm = np.where(segmentation_mask != 0)
    if len(where_worm[0]) == 0 or len(where_worm[1]) == 0:
        worm_roi = np.s_[0 : frame_copy.shape[0], 0 : frame_copy.shape[1]]
    else:
        worm_roi = np.s_[
            np.min(where_worm[0]) : np.max(where_worm[0]), np.min(where_worm[1]) : np.max(where_worm[1]),
        ]

    return frame_copy, background_color, worm_roi
