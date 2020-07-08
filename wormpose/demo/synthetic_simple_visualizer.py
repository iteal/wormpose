#!/usr/bin/env python

"""
Visualizer for the synthetic images
"""

import random
from typing import Optional, Generator
import numpy as np

from wormpose.dataset.loader import load_dataset
from wormpose.dataset.loaders.resizer import add_resizing_arguments, ResizeOptions
from wormpose.images.synthetic import SyntheticDataset
from wormpose.pose.postures_model import PosturesModel


class SyntheticSimpleVisualizer(object):
    """
    Utility class to visualize the synthetic images
    """

    def __init__(
        self,
        dataset_loader: str,
        dataset_path: str,
        postures_generator: Optional[Generator] = None,
        video_name: str = None,
        **kwargs
    ):
        resize_options = ResizeOptions(**kwargs)
        dataset = load_dataset(dataset_loader, dataset_path, resize_options=resize_options)

        if postures_generator is None:
            postures_generator = PosturesModel().generate()
        if video_name is None:
            video_name = dataset.video_names[0]

        features = dataset.features_dataset[video_name]
        self.skeletons = features.skeletons
        self.measurements = features.measurements

        self.output_image_shape = dataset.image_shape

        self.synthetic_dataset = SyntheticDataset(
            frame_preprocessing=dataset.frame_preprocessing,
            output_image_shape=self.output_image_shape,
            enable_random_augmentations=False,
        )
        skel_is_not_nan = ~np.any(np.isnan(self.skeletons), axis=(1, 2))
        self.labelled_indexes = np.where(skel_is_not_nan)[0]
        if len(self.labelled_indexes) == 0:
            raise ValueError("No template frames found in the dataset, can't generate synthetic images.")
        self.frames_dataset = dataset.frames_dataset
        self.video_name = video_name
        self.postures_generator = postures_generator

    def generate(self):
        out_image = np.empty(self.output_image_shape, dtype=np.uint8)

        with self.frames_dataset.open(self.video_name) as frames:
            while True:
                theta = next(self.postures_generator)
                random_label_index = np.random.choice(self.labelled_indexes)
                self.synthetic_dataset.generate(
                    theta=theta,
                    template_skeleton=self.skeletons[random_label_index],
                    template_frame=frames[random_label_index],
                    out_image=out_image,
                    template_measurements=self.measurements,
                )
                yield out_image, theta


def main():
    import argparse
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_loader", type=str)
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--video_name", type=str)
    parser.add_argument("--random_seed", type=int, help="Optional random seed for deterministic results")
    add_resizing_arguments(parser)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    synth_visualizer_gen = SyntheticSimpleVisualizer(**vars(args)).generate()

    while True:
        synth_image, _ = next(synth_visualizer_gen)

        cv2.imshow("synth_image", synth_image)
        cv2.waitKey()


if __name__ == "__main__":
    main()
