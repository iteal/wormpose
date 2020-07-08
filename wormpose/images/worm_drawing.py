"""
Contains the functions that deal with drawing the worm or some overlays
"""
from typing import Callable

import cv2
import numpy as np

from wormpose.pose.centerline import get_joint_indexes


def make_draw_worm_body(body_color: int = 255) -> Callable:
    """
    Functor to draw a filled worm body on an image
    as a serie of quadrilaterals with some circles at the extremities

    :param body_color: which color the worm body will be drawn (default 255)
    :return: function to call to draw a worm on an image
    """

    #  preallocates some internal data structures
    polygon = [[]] * 4
    orthogonal = np.empty((2,), dtype=np.float)
    vertices_todraw = np.empty((4, 2), dtype=int)
    cur_joint = np.empty(2, dtype=np.float)
    prev_joint = np.empty(2, dtype=np.float)

    def _draw_extremity(img, center, radius):
        cv2.circle(img, (int(center[0]), int(center[1])), int(radius), body_color, -1, 8)

    def run(worm_thickness: np.ndarray, img: np.ndarray, skeleton: np.ndarray):

        num_centerline_joints = len(worm_thickness)

        # draw two circles at the extremities
        _draw_extremity(img, skeleton[0], worm_thickness[0])
        _draw_extremity(
            img, skeleton[num_centerline_joints - 1], worm_thickness[num_centerline_joints - 1],
        )

        rx, ry = skeleton[:, 0], skeleton[:, 1]

        # draw convex quadrilateral polygons in the middle
        prev_boundary_0 = None
        prev_boundary_1 = None

        for i in range(2, num_centerline_joints):
            cur_joint[0] = rx[i]
            cur_joint[1] = ry[i]
            prev_joint[0] = rx[i - 2]
            prev_joint[1] = ry[i - 2]

            orthogonal[0] = cur_joint[1] - prev_joint[1]
            orthogonal[1] = -(cur_joint[0] - prev_joint[0])

            length = np.sqrt(orthogonal[0] * orthogonal[0] + orthogonal[1] * orthogonal[1])
            orthogonal[0] /= length
            orthogonal[1] /= length

            polygon[0] = cur_joint + orthogonal * worm_thickness[i]
            polygon[1] = cur_joint - orthogonal * worm_thickness[i]
            if prev_boundary_0 is None and prev_boundary_1 is None:
                polygon[2] = prev_joint - orthogonal * worm_thickness[i - 1]
                polygon[3] = prev_joint + orthogonal * worm_thickness[i - 1]
            else:
                polygon[3] = prev_boundary_0
                polygon[2] = prev_boundary_1

            prev_boundary_0 = polygon[0]
            prev_boundary_1 = polygon[1]

            vertices_todraw[0] = polygon[0]
            vertices_todraw[1] = polygon[1]
            vertices_todraw[2] = polygon[2]
            vertices_todraw[3] = polygon[3]
            cv2.fillConvexPoly(img, vertices_todraw, body_color, 8)

    return run


def draw_skeleton(image: np.ndarray, skeleton: np.ndarray, color, head_color):
    """
    Draw the worm centerline as lines and draw the worm head as a circle
    """
    if np.any(np.isnan(skeleton)):
        return

    skeleton = skeleton.astype(int)
    for skel_index in range(len(skeleton) - 1):
        cv2.line(
            image,
            (skeleton[skel_index][0], skeleton[skel_index][1]),
            (skeleton[skel_index + 1][0], skeleton[skel_index + 1][1]),
            color=color,
            thickness=1,
        )
    cv2.circle(image, (skeleton[0][0], skeleton[0][1]), 2, head_color, -1)


def draw_width_circle(image: np.ndarray, center, width: float, color):
    """
    Draw a circle on top of an image to represent the worm width
    """
    cv2.circle(
        image, center=(int(center[0]), int(center[1])), radius=int(width / 2), color=color,
    )


def draw_measurements(image: np.ndarray, skeleton: np.ndarray, measurements, color):
    """
    Draw overlays on top of an image to represent the worm measurements:
    circles to indicate the width at three points along the worm body: head, midbody and tail
    """
    if np.any(np.isnan(skeleton)):
        return

    head_joint, mid_body_joint, tail_body_joint = get_joint_indexes(len(skeleton))

    draw_width_circle(image, skeleton[head_joint], measurements["head_width"], color)
    draw_width_circle(image, skeleton[mid_body_joint], measurements["midbody_width"], color)
    draw_width_circle(image, skeleton[tail_body_joint], measurements["tail_width"], color)
