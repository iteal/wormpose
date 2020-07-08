#!/usr/bin/env python

"""
Evaluates a trained model by predicting new synthetic images and calculating the image similarity and the angle error
"""

import glob
import logging
import multiprocessing as mp
import os
import pickle
import random
import tempfile
from argparse import Namespace
from functools import partial
from typing import List, Optional

import numpy as np
import tensorflow as tf

from wormpose.commands import _log_parameters
from wormpose.config import default_paths
from wormpose.config.default_paths import CONFIG_FILENAME
from wormpose.config.experiment_config import load_config, add_config_argument
from wormpose.dataset import Dataset
from wormpose.dataset.loader import load_dataset, get_dataset_name
from wormpose.dataset.loaders.resizer import ResizeOptions
from wormpose.images.scoring import ResultsScoring, BaseScoringDataManager
from wormpose.machine_learning.best_models_saver import BestModels
from wormpose.machine_learning.generic_file_writer import GenericFileWriter
from wormpose.machine_learning.synthetic_data_generator import SyntheticDataGenerator
from wormpose.pose.distance_metrics import angle_distance
from wormpose.pose.eigenworms import load_eigenworms_matrix, theta_to_modes
from wormpose.pose.postures_model import PosturesModel
from wormpose.pose.results_datatypes import ShuffledResults

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _write_detailed_labeled_data_to_pickle(f, **kwargs):
    pickle.dump(
        (
            kwargs["image_data"],
            kwargs["template_measurements"],
            kwargs["template_frame"],
            kwargs["template_skeleton"],
            kwargs["theta"],
        ),
        f,
    )


class _PickleDetailedLabeledDataWriter(GenericFileWriter):
    """This labeled data writer saves detailed information about the synthetic image generation"""

    def __init__(self, filename):
        super().__init__(
            open_file=partial(open, filename, "wb"),
            write_file=lambda f, data: _write_detailed_labeled_data_to_pickle(f, **data),
        )


def _eval_data_gen(filenames: List[str]):
    for filename in filenames:
        with open(filename, "rb") as f:
            while True:
                try:
                    res = pickle.load(f)
                    im = res[0]
                    im = im[:, :, np.newaxis]
                    im = im.astype(np.float32) / 255
                    yield im
                except EOFError:
                    break


def _load_templates(pkl_filenames: List[str]):
    all_templates_data = []
    all_labels = []
    for pkl_filename in pkl_filenames:
        with open(pkl_filename, "rb") as pkl_file:
            try:
                while True:
                    (frame, template_measurements, template_frame, template_skeleton, label_theta) = pickle.load(
                        pkl_file
                    )
                    all_templates_data.append([template_frame, template_skeleton, template_measurements, frame])
                    all_labels.append(label_theta)
            except EOFError:
                pass
    return all_templates_data, all_labels


class _ScoringDataManager(BaseScoringDataManager):
    def __init__(self, pkl_filenames):
        self._all_templates_data, _ = _load_templates(pkl_filenames)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __getitem__(self, frame_index):
        return self._all_templates_data[frame_index]

    def __len__(self):
        return len(self._all_templates_data)


def _parse_arguments(dataset_path: str, kwargs: dict):
    if kwargs.get("work_dir") is None:
        kwargs["work_dir"] = default_paths.WORK_DIR
    if kwargs.get("num_process") is None:
        kwargs["num_process"] = os.cpu_count()
    if kwargs.get("temp_dir") is None:
        kwargs["temp_dir"] = tempfile.gettempdir()
    if kwargs.get("batch_size") is None:
        kwargs["batch_size"] = 512
    if kwargs.get("num_samples") is None:
        kwargs["num_samples"] = 1000
    if kwargs.get("postures_generation") is None:
        kwargs["postures_generation"] = PosturesModel().generate
    if kwargs.get("video_names") is None:
        kwargs["video_names"] = None
    if kwargs.get("model_path") is None:
        kwargs["model_path"] = None
    if kwargs.get("random_seed") is None:
        kwargs["random_seed"] = None
    if kwargs.get("eigenworms_matrix_path") is None:
        kwargs["eigenworms_matrix_path"] = None
    kwargs["temp_dir"] = tempfile.mkdtemp(dir=kwargs["temp_dir"])

    dataset_name = get_dataset_name(dataset_path)
    kwargs["experiment_dir"] = os.path.join(kwargs["work_dir"], dataset_name)

    if kwargs.get("model_path") is None:
        default_models_dir = os.path.join(kwargs["experiment_dir"], default_paths.MODELS_DIRS)
        kwargs["model_path"] = BestModels(default_models_dir).best_model_path
    if kwargs.get("config") is None:
        kwargs["config"] = os.path.join(kwargs["experiment_dir"], CONFIG_FILENAME)

    _log_parameters(logger.info, {"dataset_path": dataset_path})
    _log_parameters(logger.info, kwargs)

    return Namespace(**kwargs)


def evaluate(dataset_path: str, **kwargs):
    """
    Evaluate a trained model by predicting synthetic data and recording the image similarity

    :param dataset_path: Root path of the dataset containing videos of worm
    """
    args = _parse_arguments(dataset_path, kwargs)

    mp.set_start_method("spawn", force=True)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    results_dir = os.path.join(args.experiment_dir, "evaluation")
    os.makedirs(results_dir, exist_ok=True)

    config = load_config(args.config)
    eigenworms_matrix = load_eigenworms_matrix(args.eigenworms_matrix_path)

    dataset = load_dataset(
        dataset_loader=config.dataset_loader,
        dataset_path=dataset_path,
        selected_video_names=args.video_names,
        resize_options=ResizeOptions(resize_factor=config.resize_factor),
    )

    pkl_filenames = _generate_synthetic_data(
        dataset, args.num_process, args.num_samples, args.postures_generation, args.temp_dir, args.random_seed,
    )

    keras_model = tf.keras.models.load_model(args.model_path, compile=False)

    tf_dataset = tf.data.Dataset.from_generator(
        partial(_eval_data_gen, pkl_filenames), tf.float32, tf.TensorShape(dataset.image_shape + (1,)),
    ).batch(args.batch_size)

    network_predictions = keras_model.predict(tf_dataset)[: args.num_samples]
    shuffled_results = ShuffledResults(random_theta=network_predictions)

    ResultsScoring(
        frame_preprocessing=dataset.frame_preprocessing,
        num_process=args.num_process,
        temp_dir=args.temp_dir,
        image_shape=dataset.image_shape,
    )(
        results=shuffled_results, scoring_data_manager=_ScoringDataManager(pkl_filenames),
    )
    # Keep the maximum score between the two head/tail options for this evaluation
    image_scores = np.max(shuffled_results.scores, axis=1)

    # Now calculate the angle error and mode error
    angle_error = []
    modes_error = []
    theta_predictions = []
    _, theta_labels = _load_templates(pkl_filenames)
    for theta_label, theta_results in zip(theta_labels, shuffled_results.theta):
        dists = [angle_distance(theta_result, theta_label) for theta_result in theta_results]
        closest_index = int(np.argmin(dists))
        closest_theta = theta_results[closest_index]
        theta_predictions.append(closest_theta)
        angle_error.append(dists[closest_index])
        if eigenworms_matrix is not None:
            modes_label = theta_to_modes(theta_label, eigenworms_matrix)
            modes_prediction = theta_to_modes(closest_theta, eigenworms_matrix)
            mode_error = np.abs(modes_label - modes_prediction)
            modes_error.append(mode_error)

    np.savetxt(os.path.join(results_dir, "image_score.txt"), image_scores)
    np.savetxt(os.path.join(results_dir, "angle_error.txt"), angle_error)
    np.savetxt(os.path.join(results_dir, "theta_labels.txt"), theta_labels)
    np.savetxt(os.path.join(results_dir, "theta_predictions.txt"), theta_predictions)
    if eigenworms_matrix is not None:
        np.savetxt(os.path.join(results_dir, "modes_error.txt"), modes_error)

    logger.info(
        f"Evaluated model with synthetic data,"
        f" average image similarity: {np.mean(image_scores):.4f},"
        f" average angle error (degrees): {np.rad2deg(np.mean(angle_error)):.2f}"
    )


def _generate_synthetic_data(
    dataset: Dataset,
    num_process: int,
    num_samples: int,
    postures_generation,
    temp_dir: str,
    random_seed: Optional[int],
):
    syn_data_file_pattern = os.path.join(temp_dir, "synthetic_{index}.pkl")
    synthetic_data_generator = SyntheticDataGenerator(
        num_process=num_process,
        temp_dir=temp_dir,
        dataset=dataset,
        postures_generation_fn=postures_generation,
        writer=_PickleDetailedLabeledDataWriter,
        enable_random_augmentations=False,
        random_seed=random_seed,
    )
    gen = synthetic_data_generator.generate(num_samples=num_samples, file_pattern=syn_data_file_pattern)
    list(gen)
    pkl_filenames = list(sorted(glob.glob(syn_data_file_pattern.format(index="*"))))
    return pkl_filenames


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--model_path", type=str, help="Load models from this path.")
    parser.add_argument(
        "--video_names",
        type=str,
        nargs="+",
        help="Only evaluate using a subset of videos. " "If not set, will include all videos in dataset_path.",
    )
    parser.add_argument("--work_dir", type=str, help="Root folder for all experiments")
    parser.add_argument(
        "--num_samples", type=int, help="How many synthetic samples to evaluate the model with",
    )
    parser.add_argument(
        "--eigenworms_matrix_path", help="Path to optional eigenworms matrix to also calculate mode error",
    )
    add_config_argument(parser)
    parser.add_argument("--temp_dir", type=str, help="Where to store temporary intermediate results")
    parser.add_argument("--num_process", type=int, help="How many worker processes")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--random_seed", type=int, help="Optional random seed for deterministic results")
    args = parser.parse_args()

    evaluate(**vars(args))


if __name__ == "__main__":
    main()
