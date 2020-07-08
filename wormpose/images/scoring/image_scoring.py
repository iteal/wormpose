"""
Functions that deal with the image similarity
"""

import cv2
import numpy as np


def calculate_similarity(source_image: np.ndarray, template_image: np.ndarray):
    """
    Calculate how similar a template image is to a source image

    :param source_image: image to be compared to
    :param template_image: image to compare to the source_image
    :return: similarity score and position of the best match
    """

    # calculate cross correlation between a source and a template image
    score_map = cv2.matchTemplate(source_image, template_image, method=cv2.TM_CCOEFF_NORMED)

    # find the maximum value (and its location) of the cross correlation map
    _, score, _, score_loc = cv2.minMaxLoc(score_map)

    # we take the absolute value of the cross correlation as our image similarity
    similarity = abs(score)

    return similarity, score_loc


def fit_bounding_box_to_worm(worm_image: np.ndarray, background_color: int, padding: int = 2):
    """
    Calculates the bounding box of a worm in a real processed or a synthetic image:
    it should contain a worm object in the center and have a uniform background color
    The bounding box will be a little bigger than the worm (padding param)

    :param worm_image: image containing a worm
    :param background_color: the color of the background (should be uniform)
    :param padding: extra padding that will be added around the worm bounding box
    :return: coordinates left, right, bottom, top of the bounding box of the worm
    """
    where_mask = np.where(worm_image != background_color)
    if len(where_mask[0]) > 0 and len(where_mask[1]) > 0:
        left, bottom = np.clip(np.min(where_mask, axis=1) - padding, a_min=0, a_max=worm_image.shape)
        right, top = np.clip(np.max(where_mask, axis=1) + 1 + padding, a_min=0, a_max=worm_image.shape)
    else:
        left, right, bottom, top = 0, -1, 0, -1

    return left, right, bottom, top
