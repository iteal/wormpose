#!/usr/bin/env python

"""
Visualize the WormPose results as images showing the centerline on top of the original image
"""

import logging
import os
import shutil
import tempfile
import time
from argparse import Namespace

import cv2
import h5py
import numpy as np

from wormpose.commands import _log_parameters
from wormpose.config import default_paths
from wormpose.config.default_paths import RESULTS_FILENAME, CONFIG_FILENAME
from wormpose.config.experiment_config import load_config, add_config_argument
from wormpose.dataset.loader import get_dataset_name, Dataset
from wormpose.dataset.loader import load_dataset
from wormpose.images.worm_drawing import draw_skeleton

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class _Visualizer(object):
    def __init__(
        self, dataset: Dataset, results_dir: str, temp_dir: str, draw_original: bool,
    ):
        self.dataset = dataset
        self.results_dir = results_dir
        self.temp_dir = temp_dir
        self.draw_original = draw_original

    def export_to_images(self, video_name: str, results_scores, results_skel):

        results_dir = os.path.join(self.results_dir, video_name)

        images_temp_dir = tempfile.mkdtemp(dir=self.temp_dir)
        features = self.dataset.features_dataset[video_name]
        timestamp = features.timestamp

        image_filename_format = "time_{{:0{}d}}_frame_{{:0{}d}}_score_{{:.2f}}.png".format(
            len(str(timestamp[-1])), len(str(len(results_scores)))
        )
        start = time.time()

        original_color = (255, 255, 255)
        predicted_color = (0, 255, 0)

        with self.dataset.frames_dataset.open(video_name) as frames:

            for cur_time, (score, skel) in enumerate(zip(results_scores, results_skel)):

                frame_index = np.where(timestamp == cur_time)[0]
                if len(frame_index) == 0:
                    continue
                cur_frame_index = frame_index[0]
                cur_raw_frame = cv2.cvtColor(frames[cur_frame_index], cv2.COLOR_GRAY2BGR)

                if self.draw_original:
                    # draw original skeleton in white
                    draw_skeleton(
                        cur_raw_frame, features.skeletons[cur_frame_index], original_color, original_color,
                    )
                # draw skeleton from network predictions in green
                draw_skeleton(cur_raw_frame, skel, predicted_color, predicted_color)

                cv2.imwrite(
                    os.path.join(images_temp_dir, image_filename_format.format(cur_time, cur_frame_index, score),),
                    cur_raw_frame,
                )

        shutil.make_archive(os.path.join(results_dir, "images_results"), "zip", images_temp_dir)
        shutil.rmtree(images_temp_dir)

        end = time.time()
        logger.info(f"Exported result images for {os.path.basename(results_dir)} " f"in {end - start:.1f}s")


def _parse_arguments(dataset_path: str, kwargs: dict):

    if kwargs.get("temp_dir") is None:
        kwargs["temp_dir"] = tempfile.gettempdir()
    if kwargs.get("draw_original") is None:
        kwargs["draw_original"] = True
    if kwargs.get("work_dir") is None:
        kwargs["work_dir"] = default_paths.WORK_DIR
    if kwargs.get("video_names") is None:
        kwargs["video_names"] = None
    if kwargs.get("results_file") is None:
        kwargs["results_file"] = RESULTS_FILENAME
    if kwargs.get("group_name") is None:
        kwargs["group_name"] = "resolved"
    kwargs["temp_dir"] = tempfile.mkdtemp(dir=kwargs["temp_dir"])

    dataset_name = get_dataset_name(dataset_path)
    kwargs["experiment_dir"] = os.path.join(kwargs["work_dir"], dataset_name)

    if kwargs.get("config") is None:
        kwargs["config"] = os.path.join(kwargs["experiment_dir"], CONFIG_FILENAME)

    _log_parameters(logger.info, {"dataset_path": dataset_path})
    _log_parameters(logger.info, kwargs)

    return Namespace(**kwargs)


def visualize(dataset_path: str, **kwargs):
    """
    Export prediction results as videos with a centerline overlay on top of the original images

    :param dataset_path: Root path of the dataset containing videos of worm
    """
    args = _parse_arguments(dataset_path, kwargs)

    results_dir = os.path.join(args.experiment_dir, default_paths.RESULTS_DIR)
    config = load_config(args.config)

    dataset = load_dataset(config.dataset_loader, dataset_path, selected_video_names=args.video_names)

    visualizer = _Visualizer(
        dataset=dataset, draw_original=args.draw_original, temp_dir=args.temp_dir, results_dir=results_dir,
    )

    for video_name in dataset.video_names:

        results_file = os.path.join(results_dir, video_name, args.results_file)
        if not os.path.exists(results_file):
            logger.error(f"No results file to analyze, file not found: '{results_file}'")
            continue

        with h5py.File(results_file, "r") as f:
            if args.group_name not in f:
                logger.error(
                    f"Field: '{args.group_name}' not found in file: '{results_file}', " f"can't visualize results."
                )
                continue

            group = f[args.group_name]
            scores = group["scores"][:]
            skeletons = group["skeletons"][:]

        visualizer.export_to_images(video_name=video_name, results_scores=scores, results_skel=skeletons)

    # cleanup
    shutil.rmtree(args.temp_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--results_file", type=str, help="Path of the results file, or use default locations")
    parser.add_argument("--group_name", type=str, help="Name of the H5 group in the results file, or use default")
    parser.add_argument(
        "--video_names",
        type=str,
        nargs="+",
        help="Only analyze a subset of videos. If not set, will analyze all videos in dataset_path.",
    )
    parser.add_argument("--temp_dir", type=str, help="Where to store temporary intermediate results")
    parser.add_argument("--work_dir", type=str, help="Root folder for all experiments")
    add_config_argument(parser)
    args = parser.parse_args()

    visualize(**vars(args))


if __name__ == "__main__":
    main()
