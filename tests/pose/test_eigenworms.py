import numpy as np

import pytest

from wormpose.pose.eigenworms import load_eigenworms_matrix

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
