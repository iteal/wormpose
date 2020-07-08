"""
Utility functions to deal with eigenworms
"""

import numpy as np


def load_eigenworms_matrix(eigenworms_matrix_path: str) -> np.ndarray:
    """
    Load eigenworms matrix into numpy array from csv file

    :param eigenworms_matrix_path: path of the csv file
    :return: numpy array of the eigenworms matrix
    """
    return (
        np.loadtxt(eigenworms_matrix_path, delimiter=",").astype(float) if eigenworms_matrix_path is not None else None
    )


def theta_to_modes(theta: np.ndarray, eigenworms_matrix: np.ndarray) -> np.ndarray:
    """
    Convert angles to modes with an eigenworms matrix. We subtract the mean angle before converting.

    :param theta: angle vector, numpy array of shape (N,)
    :param eigenworms_matrix:
    :return: the modes corresponding to the angles
    """
    return eigenworms_matrix.dot(theta - np.mean(theta))


def modes_to_theta(modes: np.ndarray, eigenworms_matrix: np.ndarray) -> np.ndarray:
    """
    Convert modes to angles with an eigenworms matrix

    :param modes:
    :param eigenworms_matrix:
    :return: the angles corresponding to the modes
    """
    return modes.dot(eigenworms_matrix[:, : len(modes)].T)
