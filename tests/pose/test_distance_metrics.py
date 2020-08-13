import numpy as np

from wormpose.pose.distance_metrics import angle_distance


def test_angle_distance():

    dims = 10

    input_outputs = [
        (np.zeros(dims, dtype=float), np.zeros(dims, dtype=float), 0),
        (np.ones(dims, dtype=float) * np.pi / 6, np.zeros(dims, dtype=float), np.pi / 6),
        (np.ones(dims, dtype=float), np.ones(dims, dtype=float), 0),
        (np.ones(dims, dtype=float) * np.pi / 2, np.ones(dims, dtype=float) * np.pi, np.pi / 2),
        (np.zeros(dims, dtype=float), np.ones(dims, dtype=float) * 2 * np.pi, 0),
        (np.ones(dims, dtype=float) * np.pi / 4, np.ones(dims, dtype=float) * np.pi / 8 + 2 * np.pi, np.pi / 8),
    ]

    for input_a, input_b, expected_output in input_outputs:
        output = angle_distance(input_a, input_b)
        assert np.allclose(output, expected_output, atol=1e-5)
