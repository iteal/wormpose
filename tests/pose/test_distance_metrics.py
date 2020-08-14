import numpy as np

from wormpose.pose.distance_metrics import angle_distance, skeleton_distance

np.seterr(divide="ignore", invalid="ignore")


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


def test_skeleton_distance():

    dims = 10

    invalid_skel_1 = np.ones((dims, 2), dtype=float)
    invalid_skel_2 = invalid_skel_1 * np.nan
    straight_skel = np.array([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]).T
    straight_skel_perp = np.array(straight_skel)
    straight_skel_perp[:, 0] = -straight_skel_perp[:, 0]

    input_outputs = [
        (invalid_skel_1, invalid_skel_1, np.nan),
        (straight_skel, invalid_skel_1, np.nan),
        (straight_skel, invalid_skel_2, np.nan),
        (straight_skel, straight_skel, 1.0),
        (straight_skel, -straight_skel, -1.0),
        (straight_skel, straight_skel_perp, 0.0),
    ]

    for input_a, input_b, expected_output in input_outputs:
        output = skeleton_distance(input_a, input_b)
        assert np.allclose(output, expected_output, atol=1e-5, equal_nan=True)
