import numpy as np

import pytest

from wormpose.pose.eigenworms import load_eigenworms_matrix, modes_to_theta, theta_to_modes

_EIGENWORMS_FILE_PATH = "extras/EigenWorms.csv"


def test_load_eigenworms_file():
    eigenworms_matrix = load_eigenworms_matrix(_EIGENWORMS_FILE_PATH)

    assert type(eigenworms_matrix) is np.ndarray
    assert eigenworms_matrix.shape[0] == eigenworms_matrix.shape[1]
    assert eigenworms_matrix.dtype == float


def test_reload_eigenworms_file(tmp_path):
    eigenworms_matrix = load_eigenworms_matrix(_EIGENWORMS_FILE_PATH)

    new_file_path = tmp_path / "eigenworms.csv"
    np.savetxt(new_file_path, eigenworms_matrix, delimiter=",")

    eigenworms_matrix_reloaded = load_eigenworms_matrix(new_file_path)

    assert type(eigenworms_matrix_reloaded) == type(eigenworms_matrix)
    assert eigenworms_matrix_reloaded.shape == eigenworms_matrix.shape
    assert eigenworms_matrix_reloaded.dtype == eigenworms_matrix.dtype
    assert np.allclose(eigenworms_matrix_reloaded, eigenworms_matrix)


def test_load_none():
    eigenworms_matrix = load_eigenworms_matrix(None)
    assert eigenworms_matrix is None


def test_load_invalid_path():
    with pytest.raises(OSError):
        load_eigenworms_matrix("invalid_path")


def test_modes_to_theta():
    eigenworms_matrix = load_eigenworms_matrix(_EIGENWORMS_FILE_PATH)

    dims = eigenworms_matrix.shape[1]

    input_outputs = [
        (np.full(dims, np.nan), np.full(dims, np.nan)),
        (np.zeros(dims, dtype=float), np.zeros(dims, dtype=float)),
        (np.array([1] + [0] * (dims - 1)).astype(float), eigenworms_matrix[:, 0]),
        (np.array([0] * (dims - 1) + [-10]).astype(float), eigenworms_matrix[:, -1] * -10),
        (np.array([3] + [0] * (dims - 1)).astype(float), eigenworms_matrix[:, 0] * 3),
    ]

    for input, expected_output in input_outputs:
        output = modes_to_theta(input, eigenworms_matrix)
        assert np.allclose(output, expected_output, atol=1e-3, equal_nan=True)


def test_theta_to_modes():
    eigenworms_matrix = load_eigenworms_matrix(_EIGENWORMS_FILE_PATH)

    dims = eigenworms_matrix.shape[1]

    input_outputs = [
        (np.full(dims, np.nan), np.full(dims, np.nan)),
        (np.zeros(dims, dtype=float), np.zeros(dims, dtype=float)),
        (np.ones(dims, dtype=float), np.zeros(dims, dtype=float)),
        (eigenworms_matrix[:, 0], np.array([1] + [0] * (dims - 1)).astype(float)),
        (eigenworms_matrix[:, 1] * 2, np.array([0, 2] + [0] * (dims - 2)).astype(float)),
    ]

    for input, expected_output in input_outputs:
        output = theta_to_modes(input, eigenworms_matrix)
        assert np.allclose(output, expected_output, atol=1e-3, equal_nan=True)
