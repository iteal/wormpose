"""
Module responsible for drawing the synthetic worm image
"""

import random
from typing import Tuple

import cv2
import numpy as np

from wormpose import BaseFramePreprocessing
from wormpose.dataset.image_processing import frame_preprocessor
from wormpose.images.synthetic import (
    MAX_OFFSET_PERCENT,
    GAUSSIAN_BLUR_MIN_SIZE,
    GAUSSIAN_BLUR_MAX_SIZE_PERCENT,
    GAUSSIAN_BLUR_MAX_SIZE,
    THICKNESS_MULTIPLIER_OFFSET,
    THICKNESS_MULTIPLIER,
    WORM_LENGTH_MULTIPLIER_OFFSET,
    BLUR_PROBABILITY,
)
from wormpose.images.synthetic._coord_transform import make_calc_all_coords
from wormpose.images.synthetic._helpers import WormOutlineMask, PatchDrawing, TargetSkeletonCalculator
from wormpose.pose.centerline import get_joint_indexes


def _make_synth_worm_measurements(nb_skeleton_joints: int, measurements):
    """
    Transforms the original measurements to make them usable by the synthetic dataset generator
    The width of the worm is interpolated between the known joints (head, midbody and tail)
    """
    head_width = np.nanmean(measurements["head_width"])
    midbody_width = np.nanmean(measurements["midbody_width"])
    tail_width = np.nanmean(measurements["tail_width"])

    head_joint, mid_body_joint, tail_body_joint = get_joint_indexes(nb_skeleton_joints)
    worm_thickness = np.empty(nb_skeleton_joints, np.float32)
    worm_thickness[:head_joint] = head_width
    worm_thickness[head_joint:mid_body_joint] = np.linspace(head_width, midbody_width, mid_body_joint - head_joint)
    worm_thickness[mid_body_joint:tail_body_joint] = np.linspace(
        midbody_width, tail_width, tail_body_joint - mid_body_joint
    )
    worm_thickness[tail_body_joint:] = tail_width

    worm_length = np.nanmean(measurements["worm_length"])

    return worm_length, worm_thickness


class SyntheticDataset(object):
    """
    Class responsible for generating synthetic images
    """

    def __init__(
        self, frame_preprocessing: BaseFramePreprocessing, enable_random_augmentations: bool, output_image_shape,
    ):
        """

        :param frame_preprocessing: the FramePreprocessing object containing the image preprocessing logic
        :param enable_random_augmentations: Adds augmentation to the syntthetic images if set to True:
            translation, scale, blur
        :param output_image_shape: Desired size of the synthetic images
        """

        self.output_image_shape = (output_image_shape[0], output_image_shape[1])
        self.enable_random_augmentations = enable_random_augmentations
        self.frame_preprocessing = frame_preprocessing

        self._max_offset = (
            int(self.output_image_shape[0] * MAX_OFFSET_PERCENT),
            int(self.output_image_shape[1] * MAX_OFFSET_PERCENT),
        )
        self._nb_skeleton_joints = None
        self._theta_dims = None
        self._calc_all_coords = None
        self._synth_worm_measurements_cache = {}
        self._worm_outline_mask = WormOutlineMask(self.output_image_shape)
        self._frames_infos_cache = {}
        self._patch_drawing = PatchDrawing(output_image_shape=self.output_image_shape)
        self._target_skeleton_calculator = None
        self._target_skeleton = None
        self._gaussian_blur_filter_sizes = None

        # blurring image when random augmentations apply :
        # blur kernel size is chosen between GAUSSIAN_BLUR_MIN_SIZE and GAUSSIAN_BLUR_MAX_SIZE
        # (or a GAUSSIAN_BLUR_MAX_SIZE_PERCENT of the image size if smaller than GAUSSIAN_BLUR_MAX_SIZE)
        if self.enable_random_augmentations:
            self._gaussian_blur_filter_sizes = np.arange(
                GAUSSIAN_BLUR_MIN_SIZE,
                min(GAUSSIAN_BLUR_MAX_SIZE + 1, int(GAUSSIAN_BLUR_MAX_SIZE_PERCENT * min(output_image_shape))),
                2,
            )

        # pre allocate numpy arrays for calculations
        self._out_image = np.empty(self.output_image_shape, dtype=np.float32)

    def _get_recentered_frame(self, template_frame, template_skeleton):

        frame_id = hash(template_frame.tostring())

        if frame_id in self._frames_infos_cache:
            worm_roi, bg_mean_color = self._frames_infos_cache[frame_id]
        else:
            frame, bg_mean_color, worm_roi = frame_preprocessor.run(self.frame_preprocessing, template_frame)
            self._frames_infos_cache[frame_id] = (worm_roi, bg_mean_color)

        template_skeletons_recentered = template_skeleton - (worm_roi[1].start, worm_roi[0].start,)
        return template_frame[worm_roi], bg_mean_color, template_skeletons_recentered

    def _get_synth_measurements(self, worm_measurements):

        measurements_id = id(worm_measurements)

        if measurements_id not in self._synth_worm_measurements_cache:
            self._synth_worm_measurements_cache[measurements_id] = _make_synth_worm_measurements(
                self._nb_skeleton_joints, worm_measurements
            )

        return self._synth_worm_measurements_cache[measurements_id]

    def generate(
        self,
        theta: np.ndarray,
        template_frame: np.ndarray,
        template_skeleton: np.ndarray,
        template_measurements: np.ndarray,
        out_image: np.ndarray,
    ) -> Tuple[int, np.ndarray]:
        """
        Generates a synthetic image

        :param theta: desired posture (angles)
        :param template_frame: reference image
        :param template_skeleton: skeleton of the reference image
        :param template_measurements: measurements of the reference image (worm width and length)
        :param out_image: canvas to draw the synthetic image
        :return: background color and the synthetic worm skeleton coordinates
        """

        if out_image.dtype != np.uint8:
            raise NotImplementedError("Only uint8 images supported.")

        if self._nb_skeleton_joints is None:
            self._nb_skeleton_joints = len(template_skeleton)
            self._target_skeleton = np.empty_like(template_skeleton)
        if len(template_skeleton) != self._nb_skeleton_joints:
            raise NotImplementedError("Can't change the amount of skeleton joints.")

        if self._theta_dims is None:
            self._theta_dims = len(theta)
        if len(theta) != self._theta_dims:
            raise NotImplementedError("Can't change theta dimensions.")

        if self._target_skeleton_calculator is None:
            self._target_skeleton_calculator = TargetSkeletonCalculator(
                theta_dims=self._theta_dims,
                nb_skeletons_joints=self._nb_skeleton_joints,
                output_image_shape=self.output_image_shape,
            )
        if self._calc_all_coords is None:
            self._calc_all_coords = make_calc_all_coords(
                nb_skeleton_joints=self._nb_skeleton_joints,
                enable_random_augmentations=self.enable_random_augmentations,
            )

        # get image and skeleton recentered to region of interest
        recentered_frame, cur_bg_color, template_skel = self._get_recentered_frame(template_frame, template_skeleton)

        # get worm length and worm thickness from cache
        cur_worm_length, cur_worm_thickness = self._get_synth_measurements(template_measurements)

        # increase thickness to include more background pixels when making patches
        # (with some variation for data augmentation)
        thickness_multiplier_offset = THICKNESS_MULTIPLIER_OFFSET if self.enable_random_augmentations else 0
        thickness_multiplier = np.random.uniform(
            THICKNESS_MULTIPLIER - thickness_multiplier_offset, THICKNESS_MULTIPLIER + thickness_multiplier_offset
        )
        target_worm_thickness = cur_worm_thickness * thickness_multiplier

        target_worm_length = cur_worm_length
        # vary target worm length for data augmentation
        if self.enable_random_augmentations:
            target_worm_length = cur_worm_length * np.random.uniform(
                1.0 - WORM_LENGTH_MULTIPLIER_OFFSET, 1.0 + WORM_LENGTH_MULTIPLIER_OFFSET
            )

        # calculate the target skeleton coordinates, apply optional image augmentation
        self._target_skeleton_calculator.calculate(
            theta=theta, worm_length=target_worm_length, out=self._target_skeleton
        )
        target_skel = self._augment_skeleton(self._target_skeleton)

        # reset image canvas where we will draw the synthetic worm
        self._out_image.fill(0.0)

        # draw all the body patches: affine transform from template_coords to target_coords
        all_template_coords, all_target_coords = self._calc_all_coords(
            template_skel, target_skel, target_worm_thickness
        )
        for template_coords, target_coords in zip(all_template_coords, all_target_coords):
            self._patch_drawing.draw_patch(
                template_coords=target_coords,
                target_coords=template_coords,
                recentered_frame=recentered_frame,
                out_image=self._out_image,
            )

        # apply the worm outline mask to hide some unwanted pixels from the patch drawing
        self._worm_outline_mask.apply(
            worm_thickness=target_worm_thickness / 2, output_image=self._out_image, target_skel=target_skel,
        )

        # all zero pixels become background color #FIXME this doesn't work if the worm image has zero values
        self._out_image[self._out_image == 0] = cur_bg_color

        # median blur to smooth out the connections between patches while preserving edge features
        cv2.medianBlur(self._out_image, 3, dst=self._out_image)

        # optional image augmentation: gaussian blur
        self._augment_extra_blur(self._out_image)

        # convert final image from float to uint8
        np.copyto(out_image, self._out_image, casting="unsafe")

        return cur_bg_color, target_skel.copy()

    def _augment_skeleton(self, target_skel):

        # random translation offset
        tx, ty = (
            (
                np.random.uniform(-self._max_offset[0], self._max_offset[0]),
                np.random.uniform(-self._max_offset[1], self._max_offset[1]),
            )
            if self.enable_random_augmentations
            else (0, 0)
        )
        target_skel[:, 0] += tx
        target_skel[:, 1] += ty

        return target_skel

    def _augment_extra_blur(self, out_image):

        if self.enable_random_augmentations and np.random.uniform() > 1 - BLUR_PROBABILITY:
            ksize = random.choice(self._gaussian_blur_filter_sizes)
            cv2.GaussianBlur(out_image, (ksize, ksize), 0, dst=out_image)
