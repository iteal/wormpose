"""
 Image processing example implementation for segmenting the worm in an image
"""

from typing import Callable

import cv2
import numpy as np

# how much to crop the image to find the biggest blob
# (this is to avoid picking a biggest blob that is not the worm)
_CROP_PERCENT = 0.15


class ConstantThreshold:
    """
    Threshold function that always returns the same threshold
    """

    def __init__(self, threshold_value):
        self.threshold_value = threshold_value

    def __call__(self, frame: np.ndarray) -> int:
        return self.threshold_value


class OtsuThreshold(object):
    """
    Calculates automatic Otsu threshold on the blurred frame
    """

    def __init__(self, blur_kernel):
        """
        Creates an Otsu threshold operation with a preprocessing gaussian blur

        :param blur_kernel: Gaussian Kernel Size for the blur operation before the Otsu threshold method
            to split background and foreground. [height width]. height and width should be odd and can have different values.
        """
        self.blur_kernel = blur_kernel

    def __call__(self, frame: np.ndarray) -> int:
        blurred_frame = cv2.GaussianBlur(frame, self.blur_kernel, 0)
        blurred_frame[frame == 0] = 0
        background_threshold, _ = cv2.threshold(
            blurred_frame[blurred_frame > 0], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        return background_threshold


def segment_foreground(
    frame: np.ndarray,
    foreground_close_struct_element,
    foreground_dilate_struct_element,
    threshold_fn: Callable[[np.ndarray], int],
):
    """
    Processes a frame to isolate the object of interest (worm) from the background

    :param frame: image to process
    :param foreground_close_struct_element: morphological element to close holes in the foreground mask
    :param foreground_dilate_struct_element: morphological element to expand the foreground mask
    :param threshold_fn: function that will return the threshold to separate forefround from background in a frame
    :return: segmentation mask with values of 1 for the worm object and 0 for the background,
        and average value of the background pixels
    """

    # find the threshold to separate foreground from background
    background_threshold = threshold_fn(frame)

    # use the threshold to deduce background and foreground masks, fill in holes
    foreground_mask = (frame > 0).astype(np.uint8) * (frame < background_threshold).astype(np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, foreground_close_struct_element)
    background_mask = ((frame > 0).astype(np.uint8) - foreground_mask) > 0

    # calculate the average background color
    background_values = frame[background_mask]
    background_color = int(np.mean(background_values)) if len(background_values) > 0 else 0
    background_color = frame.dtype.type(background_color)

    # process the foreground mask to eliminate non worm objects
    # use connected components to find blobs, but focus on the center of the image to find the biggest
    # modify foreground_mask to only show the worm object
    nb_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground_mask)
    labels_crop_size = int(_CROP_PERCENT * max(foreground_mask.shape))
    labels_cropped = labels[
        labels_crop_size : foreground_mask.shape[0] - labels_crop_size,
        labels_crop_size : foreground_mask.shape[1] - labels_crop_size,
    ]

    if nb_labels == 1:
        foreground_mask.fill(0)

    foreground_objects_sizes = [len(np.where(labels_cropped == l)[0]) for l in range(1, nb_labels)]
    if len(foreground_objects_sizes) > 0:
        biggest_blob_label = np.argmax(foreground_objects_sizes) + 1
        foreground_mask[labels != biggest_blob_label] = 0

    # add a little padding to the foreground mask
    foreground_mask = cv2.dilate(foreground_mask, foreground_dilate_struct_element)

    return foreground_mask, background_color
