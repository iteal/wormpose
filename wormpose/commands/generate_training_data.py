#!/usr/bin/env python

"""
Generates the training and evaluation data from a dataset.
"""

import logging
import multiprocessing as mp
import os
import random
import shutil
import tempfile
import time
from argparse import Namespace

import numpy as np

from wormpose.commands import _log_parameters
from wormpose.config import default_paths
from wormpose.config.default_paths import (
    SYNTH_TRAIN_DATASET_NAMES,
    REAL_EVAL_DATASET_NAMES,
    CONFIG_FILENAME,
)
from wormpose.config.experiment_config import save_config, ExperimentConfig
from wormpose.dataset.loader import get_dataset_name
from wormpose.dataset.loader import load_dataset
from wormpose.dataset.loaders.resizer import add_resizing_arguments, ResizeOptions
from wormpose.machine_learning import eval_data_generator
from wormpose.machine_learning.synthetic_data_generator import SyntheticDataGenerator
from wormpose.machine_learning.tfrecord_file import TfrecordLabeledDataWriter
from wormpose.pose.postures_model import PosturesModel

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _parse_arguments(kwargs: dict):
    if kwargs.get("num_process") is None:
        kwargs["num_process"] = os.cpu_count()
    if kwargs.get("temp_dir") is None:
        kwargs["temp_dir"] = tempfile.gettempdir()
    if kwargs.get("num_train_samples") is None:
        kwargs["num_train_samples"] = int(5e5)
    if kwargs.get("num_eval_samples") is None:
        kwargs["num_eval_samples"] = int(1e4)
    if kwargs.get("work_dir") is None:
        kwargs["work_dir"] = default_paths.WORK_DIR
    if kwargs.get("postures_generation") is None:
        kwargs["postures_generation"] = PosturesModel().generate
    if kwargs.get("video_names") is None:
        kwargs["video_names"] = None
    if kwargs.get("random_seed") is None:
        kwargs["random_seed"] = None
    kwargs["temp_dir"] = tempfile.mkdtemp(dir=kwargs["temp_dir"])
    kwargs["resize_options"] = ResizeOptions(**kwargs)

    _log_parameters(logger.info, kwargs)

    return Namespace(**kwargs)


def generate(dataset_loader: str, dataset_path: str, **kwargs):
    """
    Generate synthetic images (training data) and processed real images (evaluation data)
    and save them to TFrecord files using multiprocessing

    :param dataset_loader: Name of the dataset loader, for example "tierpsy"
    :param dataset_path: Root path of the dataset containing videos of worm
    """
    _log_parameters(logger.info, {"dataset_loader": dataset_loader, "dataset_path": dataset_path})
    args = _parse_arguments(kwargs)

    mp.set_start_method("spawn", force=True)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # setup folders
    if not os.path.exists(args.work_dir):
        os.mkdir(args.work_dir)
    experiment_dir = os.path.join(args.work_dir, get_dataset_name(dataset_path))
    os.makedirs(experiment_dir, exist_ok=True)
    tfrecords_dataset_root = os.path.join(experiment_dir, default_paths.TRAINING_DATA_DIR)
    if os.path.exists(tfrecords_dataset_root):
        shutil.rmtree(tfrecords_dataset_root)

    dataset = load_dataset(
        dataset_loader=dataset_loader,
        dataset_path=dataset_path,
        resize_options=args.resize_options,
        selected_video_names=args.video_names,
    )

    start = time.time()
    synthetic_data_generator = SyntheticDataGenerator(
        num_process=args.num_process,
        temp_dir=args.temp_dir,
        dataset=dataset,
        postures_generation_fn=args.postures_generation,
        enable_random_augmentations=True,
        writer=TfrecordLabeledDataWriter,
        random_seed=args.random_seed,
    )
    gen = synthetic_data_generator.generate(
        num_samples=args.num_train_samples, file_pattern=os.path.join(args.temp_dir, SYNTH_TRAIN_DATASET_NAMES),
    )
    for progress in gen:
        yield progress
    yield 1.0

    theta_dims = len(next(args.postures_generation()))
    num_eval_samples = eval_data_generator.generate(
        dataset=dataset,
        num_samples=args.num_eval_samples,
        theta_dims=theta_dims,
        file_pattern=os.path.join(args.temp_dir, REAL_EVAL_DATASET_NAMES),
    )

    shutil.copytree(args.temp_dir, tfrecords_dataset_root)
    save_config(
        ExperimentConfig(
            dataset_loader=dataset_loader,
            image_shape=dataset.image_shape,
            theta_dimensions=theta_dims,
            num_train_samples=args.num_train_samples,
            num_eval_samples=num_eval_samples,
            resize_factor=args.resize_options.resize_factor,
            video_names=dataset.video_names,
        ),
        os.path.join(experiment_dir, CONFIG_FILENAME),
    )

    end = time.time()
    logger.info(f"Done generating training data in : {end - start:.1f}s")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_loader", type=str)
    parser.add_argument("dataset_path", type=str)
    parser.add_argument(
        "--video_names",
        type=str,
        nargs="+",
        help="Only generate training data for a subset of videos. "
        "If not set, will include all videos in dataset_path.",
    )
    parser.add_argument("--num_train_samples", type=int, help="How many training samples to generate")
    parser.add_argument("--num_eval_samples", type=int, help="How many evaluation samples to generate")
    parser.add_argument("--temp_dir", type=str, help="Where to store temporary intermediate results")
    parser.add_argument("--work_dir", type=str, help="Root folder for all experiments")
    parser.add_argument("--num_process", type=int, help="How many worker processes")
    parser.add_argument("--random_seed", type=int, help="Optional random seed for deterministic results")
    add_resizing_arguments(parser)
    args = parser.parse_args()

    last_progress = None
    for progress in generate(**vars(args)):
        prog_percent = int(progress * 100)
        if prog_percent != last_progress:
            logger.info(f"Generating training data: {prog_percent}% done")
        last_progress = prog_percent


if __name__ == "__main__":
    main()
