import numpy as np

from wormpose.pose.centerline import (
    flip_theta,
    flip_theta_series,
    get_joint_indexes,
    skeleton_to_angle,
    skeletons_to_angles,
    interpolate_skeleton,
)


def test_flip_theta():

    dims = 10

    input_outputs = [
        (np.full(dims, np.nan), np.full(dims, np.nan)),
        (np.zeros(dims, dtype=float), np.full(dims, np.pi)),
        (np.full(dims, np.pi / 2), np.full(dims, 3 * np.pi / 2)),
    ]

    for input, expected_output in input_outputs:
        output = flip_theta(input)
        assert np.allclose(output, expected_output, atol=1e-5, equal_nan=True)


def test_flip_theta_series():

    dims = (5, 10)

    input_outputs = [
        (np.full(dims, np.nan), np.full(dims, np.nan)),
        (np.zeros(dims, dtype=float), np.full(dims, np.pi)),
        (np.full(dims, np.pi / 2), np.full(dims, 3 * np.pi / 2)),
    ]

    for input, expected_output in input_outputs:
        output = flip_theta_series(input)
        assert np.allclose(output, expected_output, atol=1e-5, equal_nan=True)


def test_get_joint_indexes():

    nb_skeleton_joints = 10
    head_joint, midbody_joint, tail_joint = get_joint_indexes(nb_skeleton_joints)

    assert type(head_joint) == type(midbody_joint) == type(tail_joint) == int
    assert head_joint <= midbody_joint <= tail_joint <= nb_skeleton_joints


def test_skeleton_to_angle():
    dims = 10

    straight_skel = np.array([np.linspace(0, 1, dims), np.linspace(0, 1, dims)]).T
    straight_skel2 = np.array(straight_skel)
    straight_skel2[:, 0] = 0

    input_outputs = [
        (np.full((dims, 2), np.nan), np.full(dims, np.nan)),
        (straight_skel, np.full(dims, np.pi / 4)),
        (-straight_skel, np.full(dims, np.pi / 4 + np.pi)),
        (straight_skel2, np.full(dims, np.pi / 2)),
    ]

    for input, expected_output in input_outputs:
        output = skeleton_to_angle(input, dims)
        assert np.allclose(np.sin(output), np.sin(expected_output), atol=1e-5, equal_nan=True)
        assert np.allclose(np.cos(output), np.cos(expected_output), atol=1e-5, equal_nan=True)


def test_skeletons_to_angle():
    dims = 10
    num_skels = 5

    straight_skel = np.tile(np.array([np.linspace(0, 1, dims), np.linspace(0, 1, dims)]).T, (num_skels, 1, 1))
    straight_skel2 = np.array(straight_skel)
    straight_skel2[:, :, 0] = 0

    input_outputs = [
        (np.full((num_skels, dims, 2), np.nan), np.full((num_skels, dims), np.nan)),
        (straight_skel, np.full((num_skels, dims), np.pi / 4)),
        (-straight_skel, np.full((num_skels, dims), np.pi / 4 + np.pi)),
        (straight_skel2, np.full((num_skels, dims), np.pi / 2)),
    ]

    for input, expected_output in input_outputs:
        output = skeletons_to_angles(input, dims)
        assert np.allclose(np.sin(output), np.sin(expected_output), atol=1e-5, equal_nan=True)
        assert np.allclose(np.cos(output), np.cos(expected_output), atol=1e-5, equal_nan=True)


def test_interpolate_skeleton():
    dims = 10

    input_outputs = [
        (np.full((dims, 2), np.nan), 30 - 1, np.full((30, 2), np.nan)),
        (
            np.array([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]).T,
            10 - 1,
            np.array([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]).T,
        ),
        (
            np.array([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]).T,
            20 - 1,
            np.array([np.linspace(0, 1, 20), np.linspace(0, 1, 20)]).T,
        ),
        (
            np.array([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]).T,
            5 - 1,
            np.array([np.linspace(0, 1, 5), np.linspace(0, 1, 5)]).T,
        ),
    ]

    for input_a, input_b, expected_output in input_outputs:
        output = interpolate_skeleton(input_a, input_b)
        assert np.allclose(expected_output, output, atol=1e-5, equal_nan=True)
