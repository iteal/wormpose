import math
import numpy as np

from wormpose.images.synthetic import BODY_SEGMENTS_DIVIDER, DRAW_TAIL_ON_TOP_PROBABILITY


def _make_calc_one_segment_coords():
    """
    Functor to calculate patches coordinates:
    From two centerline joint coordinates, and a known width, it will calculate
     the coordinates of a rectangle aligned in between the two joints positions.
    This rectangle represents a little segment along the worm body.
    """

    # preallocates some internal data structures
    skel_perp = np.empty((2,), dtype=np.float32)

    def run(joint_a, joint_b, width, dest):
        perp = [-(joint_b[1] - joint_a[1]), joint_b[0] - joint_a[0]]
        norm = math.sqrt(perp[0] * perp[0] + perp[1] * perp[1])

        skel_perp[0] = (perp[0] / norm) * (width / 2)
        skel_perp[1] = (perp[1] / norm) * (width / 2)

        dest[0] = joint_a + skel_perp
        dest[1] = joint_b + skel_perp
        dest[2] = joint_b - skel_perp
        dest[3] = joint_a - skel_perp

    return run


def make_calc_all_coords(nb_skeleton_joints: int, enable_random_augmentations: bool):
    """
    Calculates the coordinates of all the square patches along the worm body,
    for the template posture and the target posture, so we can do the affine transform later.
    The order of the coordinates list will define their drawing order

    :param nb_skeleton_joints: How many joints has the template skeleton, will be used to define how many joints
        to include in a patch. The size of a patch is defined by BODY_SEGMENTS_DIVIDER so it is consistent
        for different datasets with different number of skeleton joints.
    :param enable_random_augmentations: If True: Will randomly reverse the order of the list of patches
        depending on DRAW_TAIL_ON_TOP_PROBABILITY. If False, the patch corresponding to the head will always
        be last in the list so that it can be drawn last (head on top of tail)
    :return: tuple (list of template patches coordinates, list of target patches coordinates)
    """
    step = max(1, nb_skeleton_joints // BODY_SEGMENTS_DIVIDER)
    last = nb_skeleton_joints - step

    template_coords = np.empty((last + 2, 4, 2), dtype=np.float32)
    target_coords = np.empty((last + 2, 4, 2), dtype=np.float32)
    calc_one_segment_coords = _make_calc_one_segment_coords()

    def _update_coords(
        width, index_to_update, template_joint_a, template_joint_b, target_joint_a, target_joint_b,
    ):
        calc_one_segment_coords(
            joint_a=template_joint_a, joint_b=template_joint_b, width=width, dest=template_coords[index_to_update],
        )
        calc_one_segment_coords(
            joint_a=target_joint_a, joint_b=target_joint_b, width=width, dest=target_coords[index_to_update],
        )

    def run(template_skel, target_skel, target_worm_thickness):
        # first calculate the patches for the two extremities (beyond head and tail)
        # to avoid to cut the synthetic worm abruptly
        # they are first in the list, so that they get drawn first (will be the most in the background)

        # before head
        _update_coords(
            width=target_worm_thickness[0],
            template_joint_a=template_skel[0] + template_skel[0] - template_skel[step],
            template_joint_b=template_skel[0],
            target_joint_a=target_skel[0] + target_skel[0] - target_skel[step],
            target_joint_b=target_skel[0],
            index_to_update=0,
        )
        # after tail
        _update_coords(
            width=target_worm_thickness[last],
            template_joint_a=template_skel[last - 1 + step],
            template_joint_b=template_skel[last - 1 + step] + template_skel[last - 1 + step] - template_skel[last - 1],
            target_joint_a=target_skel[last - 1 + step],
            target_joint_b=target_skel[last - 1 + step] + target_skel[last - 1 + step] - target_skel[last - 1],
            index_to_update=1,
        )

        # decide about the order of the patches along the worm body
        if enable_random_augmentations and np.random.random() >= DRAW_TAIL_ON_TOP_PROBABILITY:
            # draw the worm starting from the head (the tail will be on top)
            joint_indexes = range(0, last)
        else:
            # draw starting from the tail (the head will be on top)
            joint_indexes = reversed(range(0, last))

        # calculate all the other patches in the body, in head to tail or tail to head order
        for index, joint_index in enumerate(joint_indexes):
            _update_coords(
                width=target_worm_thickness[joint_index],
                template_joint_a=template_skel[joint_index],
                template_joint_b=template_skel[joint_index + step],
                target_joint_a=target_skel[joint_index],
                target_joint_b=target_skel[joint_index + step],
                index_to_update=index + 2,
            )

        return template_coords, target_coords

    return run
