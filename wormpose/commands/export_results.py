#!/usr/bin/env python

"""
Exporting WormPose results to another format, for example the WCON format for a Tierpsy dataset
"""

import glob
import logging
import os
from argparse import Namespace

import h5py

from wormpose.commands import _log_parameters
from wormpose.config import default_paths
from wormpose.config.default_paths import RESULTS_FILENAME, CONFIG_FILENAME
from wormpose.config.experiment_config import load_config, add_config_argument
from wormpose.dataset import get_dataset_name, load_dataset

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _parse_arguments(dataset_path: str, kwargs: dict):
    if kwargs.get("work_dir") is None:
        kwargs["work_dir"] = default_paths.WORK_DIR
    if kwargs.get("video_names") is None:
        kwargs["video_names"] = None

    dataset_name = get_dataset_name(dataset_path)
    kwargs["experiment_dir"] = os.path.join(kwargs["work_dir"], dataset_name)

    if kwargs.get("config") is None:
        kwargs["config"] = os.path.join(kwargs["experiment_dir"], CONFIG_FILENAME)

    _log_parameters(logger.info, {"dataset_path": dataset_path})
    _log_parameters(logger.info, kwargs)

    return Namespace(**kwargs)


def export(dataset_path: str, **kwargs):
    """
    Export WormPose results into another format, depending on the dataset

    :param dataset_path: Root path of the dataset containing videos of worm
    """
    args = _parse_arguments(dataset_path, kwargs)

    results_root_dir = os.path.join(args.experiment_dir, default_paths.RESULTS_DIR)
    results_files = glob.glob(os.path.join(results_root_dir, "*", RESULTS_FILENAME))

    if len(results_files) == 0:
        raise FileNotFoundError(f"No results files found in path: '{results_root_dir}'")

    config = load_config(args.config)

    dataset = load_dataset(config.dataset_loader, dataset_path, selected_video_names=args.video_names)

    for results_file in results_files:
        try:
            with h5py.File(results_file, "r") as f:
                results_skeletons = f["resolved"]["skeletons"][:]
        except Exception:
            logger.error(f"Couldn't read results in file {results_file}")
            continue

        video_name = os.path.basename(os.path.dirname(results_file))
        logger.info(f'Exporting: "{video_name}"')

        dataset.results_exporter.export(
            dataset=dataset,
            video_name=video_name,
            results_skeletons=results_skeletons,
            out_dir=os.path.dirname(results_file),
        )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--work_dir", type=str, help="Root folder for all experiments")
    parser.add_argument(
        "--video_names",
        type=str,
        nargs="+",
        help="Only analyze a subset of videos. If not set, will analyze all videos in dataset_path.",
    )
    add_config_argument(parser)
    args = parser.parse_args()

    export(**vars(args))


if __name__ == "__main__":
    main()
