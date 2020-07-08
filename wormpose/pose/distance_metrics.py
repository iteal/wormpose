"""
Contains function to calculate distances between worm poses, either represented as angles or as skeletons
"""

import math

import numpy as np


def angle_distance(theta_a: np.ndarray, theta_b: np.ndarray) -> float:
    """
    Angle distance that takes into account the periodicity of angles
    """
    diff = np.abs(np.arctan2(np.sin(theta_a - theta_b), np.cos(theta_a - theta_b)))
    return diff.mean()


def _head_tail_diff(skel):
    return skel[-1][0] - skel[0][0], skel[-1][1] - skel[0][1]


def _cos_similarity(a, b):
    def _norm(x):
        return math.sqrt(x[0] * x[0] + x[1] * x[1])

    return (a[0] * b[0] + a[1] * b[1]) / (_norm(a) * _norm(b))


def skeleton_distance(skel_a: np.ndarray, skel_b: np.ndarray) -> float:
    """
    Cosine similarity between the two head to tail vectors of the input skeletons
    """
    return _cos_similarity(_head_tail_diff(skel_a), _head_tail_diff(skel_b))
