"""
This module contains function related to the worm centerline, for example to convert skeleton (x, y) coordinates
to angles, flip the head-tail orientation etc
"""
from typing import Tuple, Optional

import numpy as np
from scipy.interpolate import interp1d


def calculate_skeleton(
    theta: np.ndarray,
    worm_length: float,
    out: Optional[np.ndarray] = None,
    canvas_width_height: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Calculates (x,y) coordinates of the worm centerline from the tangent angle (theta) vector

    :param theta: centerline angles
    :param worm_length: desired worm length
    :param out: optional numpy array of shape (len(theta), 2) to store the resulting skeleton
    :param canvas_width_height: optional canvas to recenter the skeleton coordinates in the middle
    :return: output skeleton: numpy array of shape (len(theta), 2) representing the centerline angles
    """
    num_centerline_joints = len(theta)

    if out is None:
        out = np.empty((num_centerline_joints, 2))

    centerline_section_length = worm_length / (num_centerline_joints - 1)
    np.cumsum(
        [centerline_section_length * np.cos(theta), centerline_section_length * np.sin(theta),], axis=1, out=out.T,
    )

    # optional center the skeleton coordinates in the middle of an image canvas
    if canvas_width_height is not None:
        out += -np.min(out, axis=0) + (canvas_width_height - (np.max(out, axis=0) - np.min(out, axis=0))) / 2

    return out


def interpolate_skeleton(skeleton: np.ndarray, new_dims: int) -> np.ndarray:
    """
    Interpolates a worm skeleton to have a different number of points
    """
    new_pos_dim = []
    for dim in range(skeleton.shape[1]):
        y = skeleton[:, dim]
        x = np.arange(y.size)
        if np.any(np.isnan(y)):
            new_pos_dim.append([np.nan] * (new_dims + 1))
        else:
            # Interpolate the data using a cubic spline to "new_length" samples
            new_length = new_dims + 1
            new_x = np.linspace(x.min(), x.max(), new_length)
            new_y = interp1d(x, y, kind="cubic")(new_x)
            new_pos_dim.append(new_y)
    new_pos = np.vstack(new_pos_dim).T
    return new_pos


def skeletons_to_angles(skeletons: np.ndarray, theta_dims: int) -> np.ndarray:
    new_skeletons = []
    for frame in range(skeletons.shape[0]):
        skeleton = skeletons[frame]
        new_skeletons.append(interpolate_skeleton(skeleton, theta_dims))
    new_skeletons = np.array(new_skeletons, skeletons.dtype)

    skel_x = new_skeletons[:, :, 0]
    skel_y = new_skeletons[:, :, 1]
    d_x = np.diff(skel_x, axis=1)
    d_y = np.diff(skel_y, axis=1)
    # calculate tangent angles.  atan2 uses angles from -pi to pi
    angles = np.arctan2(d_y, d_x)
    return angles.astype(np.float32)


def skeleton_to_angle(skeleton: np.ndarray, theta_dims: int):
    new_skeleton = interpolate_skeleton(skeleton, theta_dims)
    skel_x = new_skeleton[:, 0]
    skel_y = new_skeleton[:, 1]
    d_x = np.diff(skel_x)
    d_y = np.diff(skel_y)
    # calculate tangent angles.  atan2 uses angles from -pi to pi
    angles = np.arctan2(d_y, d_x)
    return angles.astype(np.float32)


def get_joint_indexes(nb_skeleton_joints: int) -> Tuple[int, int, int]:
    head_joint = int(0.1 * nb_skeleton_joints)
    mid_body_joint = int(0.5 * nb_skeleton_joints)
    tail_body_joint = int(0.9 * nb_skeleton_joints)

    return head_joint, mid_body_joint, tail_body_joint


def flip_theta_series(theta_series: np.ndarray) -> np.ndarray:
    """
     head-tail flip for each value in a serie
    """
    return np.flip(theta_series, axis=1) + np.pi


def flip_theta(theta: np.ndarray) -> np.ndarray:
    """
     head-tail flip for one value
    """
    return np.flip(theta, axis=0) + np.pi
