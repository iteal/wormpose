#!/usr/bin/env python

"""
Calculates the image similarity on a random selection of labeled frames from a dataset.
"""

import logging
import os
import random
from argparse import Namespace
from typing import Tuple

import h5py
import numpy as np

from wormpose.commands import _log_parameters
from wormpose.config import default_paths
from wormpose.dataset import Dataset
from wormpose.dataset.loader import get_dataset_name
from wormpose.dataset.loader import load_dataset
from wormpose.images.scoring.centerline_accuracy_check import CenterlineAccuracyCheck
from wormpose.pose.centerline import skeletons_to_angles
from wormpose.dataset.loaders.resizer import add_resizing_arguments, ResizeOptions

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class _ScoresWriter(object):
    def __init__(self):
        self.all_scores = []

    def add(self, kwargs):
        self.all_scores.append(kwargs["score"])

    def write(self, results_file):
        with h5py.File(results_file, "a") as f:
            f.create_dataset("scores", data=self.all_scores)


class _ImagesAndScoresWriter(_ScoresWriter):
    def __init__(self):
        self.all_synth = []
        self.all_real = []
        super().__init__()

    def add(self, kwargs):
        centerline_accuracy: CenterlineAccuracyCheck = kwargs["centerline_accuracy"]
        self.all_real.append(np.array(centerline_accuracy.last_real_image))
        self.all_synth.append(np.array(centerline_accuracy.last_synth_image))
        super().add(kwargs)

    def write(self, results_file):
        with h5py.File(results_file, "a") as f:
            f.create_dataset("real_images", data=self.all_real)
            f.create_dataset("synth_images", data=self.all_synth)
        super().write(results_file)


class _Calibrator(object):
    def __init__(
        self, dataset: Dataset, results_dir: str, image_shape: Tuple[int, int], num_samples: int, theta_dims: int,
    ):
        self.dataset = dataset
        self.results_dir = results_dir
        self.image_shape = image_shape
        self.num_samples = num_samples
        self.theta_dims = theta_dims

    def __call__(self, video_name: str, writer: _ScoresWriter):
        """
        Evaluate image metric score on labelled frames.
        """
        features = self.dataset.features_dataset[video_name]
        labelled_thetas = skeletons_to_angles(features.skeletons, theta_dims=self.theta_dims)
        labelled_indexes = features.labelled_indexes

        centerline_accuracy = CenterlineAccuracyCheck(
            frame_preprocessing=self.dataset.frame_preprocessing, image_shape=self.image_shape,
        )

        with self.dataset.frames_dataset.open(video_name) as frames:
            frames_amount = min(self.num_samples, len(labelled_indexes))

            random_label_index = np.random.choice(labelled_indexes, frames_amount, replace=False)
            thetas = labelled_thetas[random_label_index]

            for theta, index in zip(thetas, random_label_index):
                cur_frame = frames[index]

                score, _ = centerline_accuracy(
                    theta=theta,
                    template_skeleton=features.skeletons[index],
                    template_measurements=features.measurements,
                    template_frame=cur_frame,
                    real_frame_orig=cur_frame,
                )
                writer.add(locals())

        results_file = os.path.join(self.results_dir, video_name + "_calibration.h5")
        if os.path.exists(results_file):
            os.remove(results_file)
        writer.write(results_file=results_file)

        logger.info(
            f"Evaluated known skeletons reconstruction for {video_name},"
            f" average score {np.mean(writer.all_scores):.4f}"
        )
        return results_file


def _parse_arguments(kwargs: dict):
    if kwargs.get("num_samples") is None:
        kwargs["num_samples"] = 500
    if kwargs.get("work_dir") is None:
        kwargs["work_dir"] = default_paths.WORK_DIR
    if kwargs.get("theta_dims") is None:
        kwargs["theta_dims"] = 100
    if kwargs.get("video_names") is None:
        kwargs["video_names"] = None
    if kwargs.get("save_images") is None:
        kwargs["save_images"] = False
    if kwargs.get("random_seed") is None:
        kwargs["random_seed"] = None
    kwargs["resize_options"] = ResizeOptions(**kwargs)

    _log_parameters(logger.info, kwargs)
    return Namespace(**kwargs)


def calibrate(dataset_loader: str, dataset_path: str, **kwargs):
    """
    Calculate the image score for a certain number of labelled frames in the dataset,
    this will give an indication on choosing the image similarity threshold when predicting all frames in the dataset.

    :param dataset_loader: Name of the dataset loader, for example "tierpsy"
    :param dataset_path: Root path of the dataset containing videos of worm
    """
    _log_parameters(logger.info, {"dataset_loader": dataset_loader, "dataset_path": dataset_path})
    args = _parse_arguments(kwargs)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    dataset_name = get_dataset_name(dataset_path)
    experiment_dir = os.path.join(args.work_dir, dataset_name)
    calibration_results_dir = os.path.join(experiment_dir, default_paths.CALIBRATION_RESULTS_DIR)
    os.makedirs(calibration_results_dir, exist_ok=True)

    dataset = load_dataset(
        dataset_loader, dataset_path, resize_options=args.resize_options, selected_video_names=args.video_names
    )

    calibrator = _Calibrator(
        dataset=dataset,
        results_dir=calibration_results_dir,
        image_shape=dataset.image_shape,
        num_samples=args.num_samples,
        theta_dims=args.theta_dims,
    )

    writer = _ImagesAndScoresWriter() if kwargs["save_images"] else _ScoresWriter()

    for video_name in dataset.video_names:
        results_file = calibrator(video_name=video_name, writer=writer)
        yield video_name, results_file


def main():
    import argparse

    parser = argparse.ArgumentParser()

    # inputs
    parser.add_argument("dataset_loader", type=str)
    parser.add_argument("dataset_path", type=str)
    add_resizing_arguments(parser)
    parser.add_argument(
        "--video_names",
        type=str,
        nargs="+",
        help="Only evaluate using a subset of videos. " "If not set, will include all videos in dataset_path.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="How many frames to perform the calibration in order to evaluate the image metric",
    )
    parser.add_argument("--theta_dims", type=int)
    parser.add_argument("--work_dir", type=str, help="Root folder for all experiments")
    parser.add_argument(
        "--save_images", default=False, action="store_true", help="Also save the images used for the calibration"
    )
    parser.add_argument("--random_seed", type=int, help="Optional random seed for deterministic results")

    args = parser.parse_args()

    list(calibrate(**vars(args)))


if __name__ == "__main__":
    main()
