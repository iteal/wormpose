"""
Assesses the accuracy of a centerline theta compared to a real image, by calculating the image similarity metric
"""

import numpy as np

from wormpose import BaseFramePreprocessing
from wormpose.images.scoring import image_scoring
from wormpose.images.scoring.image_scoring import fit_bounding_box_to_worm
from wormpose.images.real_dataset import RealDataset
from wormpose.images.synthetic import SyntheticDataset


class CenterlineAccuracyCheck(object):
    """
    Class that performs comparison (image similarity) to assess how a centerline accurately represents a real image.

    It first preprocesses the original real image with a BaseFramePreprocessing class
    (crop and set the background pixels to a uniform color).
    Then, it creates a synthetic image representing the centerline theta, using a provided template image.
    Typically, the template image was chosen to be the closest labelled image in time to the real image.
    The synthetic image is cropped to fit the worm, in order to apply a template matching function between
    the real image (full size) and the synthetic image (smaller)
    The result is an image similarity value and the synthetic image skeleton coordinates.
    """

    def __init__(self, frame_preprocessing: BaseFramePreprocessing, image_shape):

        self.real_dataset = RealDataset(frame_preprocessing=frame_preprocessing, output_image_shape=image_shape)

        self.synthetic_dataset = SyntheticDataset(
            frame_preprocessing=frame_preprocessing, output_image_shape=image_shape, enable_random_augmentations=False,
        )
        self.last_synth_image = np.empty(image_shape, np.uint8)
        self.last_real_image = None

    def __call__(
        self, theta, template_skeleton, template_frame, template_measurements, real_frame_orig,
    ):
        if np.any(np.isnan(theta)):
            score = np.nan
            synth_skel = np.full_like(template_skeleton, np.nan)
            return score, synth_skel

        self.last_real_image, skel_offset = self.real_dataset.process_frame(real_frame_orig)
        cur_bg_color, synth_skel = self.synthetic_dataset.generate(
            theta,
            template_frame=template_frame,
            template_skeleton=template_skeleton,
            out_image=self.last_synth_image,
            template_measurements=template_measurements,
        )

        # Crop the synthetic image to the object of interest before doing the image comparison,
        # we don't need the full image with all the background, still keep a little padding around the worm.
        left, right, bottom, top = fit_bounding_box_to_worm(self.last_synth_image, cur_bg_color)
        np.subtract(synth_skel, (bottom, left), out=synth_skel)
        cropped_synth_image = self.last_synth_image[left:right, bottom:top]

        # Perform the image comparison between the real image and the reconstructed synthetic image cropped.
        # This gives a heat map, we are interested in the maximum value of the heatmap and its location.
        # This maximum score gives an estimation of the confidence of the prediction.
        score, score_loc = image_scoring.calculate_similarity(
            source_image=self.last_real_image, template_image=cropped_synth_image
        )

        # Using the heatmap maximum coordinates, we can transform the coordinates of the reconstructed skeleton
        # to the original image coordinates system.
        synth_skel += score_loc
        synth_skel += np.array([skel_offset[0], skel_offset[1]])

        return score, synth_skel
