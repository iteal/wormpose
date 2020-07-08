"""
Builds the real dataset, where images from a dataset are preprocessed and then cropped to be the same size
"""

from typing import Tuple

import numpy as np

from wormpose import BaseFramePreprocessing
from wormpose.dataset.image_processing import frame_preprocessor


class RealDataset(object):
    """
    The RealDataset takes a raw image and apply the FramePreprocessing, it also ensures all the resulting images
    have the same size, by expanding the region if necessary, or by simply cropping
    """

    def __init__(
        self, frame_preprocessing: BaseFramePreprocessing, output_image_shape: Tuple[int, int],
    ):
        """

        :param frame_preprocessing: the FramePreprocessing object containing the image preprocessing logic
        :param output_image_shape: the desired output size of the images
        """

        if output_image_shape[0] % 2 == 1 or output_image_shape[1] % 2 == 1:
            raise NotImplementedError(
                f"Image width and height should be even numbers: {output_image_shape}, " f"odd numbers not supported."
            )

        self.output_image_shape = np.array((output_image_shape[0], output_image_shape[1]))
        self.frame_preprocessing = frame_preprocessing

    def process_frame(self, cur_frame: np.ndarray):
        """
        Processes one image

        :param cur_frame: Raw frame from a dataset to process
        :return: Processed version of the frame and normalized to the desired image size
        """

        processed_frame, bg_mean_color, worm_roi = frame_preprocessor.run(self.frame_preprocessing, cur_frame)

        center = (
            (worm_roi[0].start + worm_roi[0].stop) // 2,
            (worm_roi[1].start + worm_roi[1].stop) // 2,
        )

        # simple case: just crop the desired shape centered around the region of interest
        if _can_crop_simply(center, self.output_image_shape, processed_frame):
            roi_coord, skel_offset = _simple_crop(center, self.output_image_shape)

        # if we can't make a simple crop :
        # the region of interest of the worm is bigger than the result image or too much on the edge
        # let's copy what fits to a new empty image
        else:
            empty_frame = np.full(self.output_image_shape, bg_mean_color, dtype=processed_frame.dtype)
            processed_frame, roi_coord, skel_offset = _complex_crop(
                empty_frame=empty_frame,
                center=center,
                out_shape=self.output_image_shape,
                processed_frame=processed_frame,
                worm_roi=worm_roi,
            )

        roi = processed_frame[roi_coord]

        return roi, skel_offset


def _can_crop_simply(center, output_image_shape: np.ndarray, processed_frame):
    """
    We can make a simple crop if the output_image_shape with its center position set, fits in the processed_frame
    """
    return (center - output_image_shape // 2 >= 0).all() and (
        center + output_image_shape // 2 < processed_frame.shape
    ).all()


def _simple_crop(center, out_shape: np.ndarray):
    top_left = (center - out_shape / 2).astype(int)
    bottom_right = (center + out_shape / 2).astype(int)

    roi_coord = np.s_[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]]

    # skeleton is in XY coordinates (numpy is YX)
    skel_offset = (top_left[1], top_left[0])

    return roi_coord, skel_offset


def _complex_crop(empty_frame, center, out_shape: np.ndarray, processed_frame, worm_roi: Tuple[slice, slice]):
    """
    Creates a new uniform image of the shape out_shape (pixels are set to bg_mean_color)
    and set what fits of the processed frame in the center
    """
    out_height, out_width = out_shape

    roi_height, roi_width = (
        worm_roi[0].stop - worm_roi[0].start,
        worm_roi[1].stop - worm_roi[1].start,
    )
    y, x = max(0, (out_height - roi_height) // 2), max(0, (out_width - roi_width) // 2)
    zone_to_copy = np.s_[y : min(y + roi_height, out_height), x : min(x + roi_width, out_width)]

    if roi_height > out_height or roi_width > out_width:
        new_roi_height = zone_to_copy[0].stop - zone_to_copy[0].start
        new_roi_width = zone_to_copy[1].stop - zone_to_copy[1].start
        y, x = center[0] - new_roi_height // 2, center[1] - new_roi_width // 2
        worm_roi = np.s_[y : y + new_roi_height, x : x + new_roi_width]

    empty_frame[zone_to_copy] = processed_frame[worm_roi]
    roi_coord = np.s_[:out_height, :out_width]
    skel_offset = (
        worm_roi[1].start - zone_to_copy[1].start,
        worm_roi[0].start - zone_to_copy[0].start,
    )

    return empty_frame, roi_coord, skel_offset
