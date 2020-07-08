#!/usr/bin/env python

"""
Predicts videos using a trained model
"""

import logging
import multiprocessing as mp
import os
import random
import shutil
import tempfile
from argparse import Namespace
from functools import partial
from typing import Tuple

import numpy as np
import tensorflow as tf

from wormpose.commands import _log_parameters
from wormpose.commands.utils.results_saver import ResultsSaver
from wormpose.commands.utils.time_sampling import resample_results
from wormpose.config import default_paths
from wormpose.config.default_paths import RESULTS_FILENAME, CONFIG_FILENAME
from wormpose.config.experiment_config import load_config, add_config_argument
from wormpose.dataset.features import Features
from wormpose.dataset.loader import get_dataset_name
from wormpose.dataset.loader import load_dataset
from wormpose.dataset.loaders.resizer import ResizeOptions
from wormpose.images.scoring import BaseScoringDataManager, ScoringDataManager, ResultsScoring
from wormpose.machine_learning.best_models_saver import BestModels
from wormpose.machine_learning.predict_data_generator import PredictDataGenerator
from wormpose.pose.centerline import skeletons_to_angles
from wormpose.pose.headtail_resolution import resolve_head_tail
from wormpose.pose.results_datatypes import (
    ShuffledResults,
    OriginalResults,
    BaseResults,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
tf.get_logger().setLevel(logging.INFO)


def _make_tf_dataset(data_generator, batch_size: int, image_shape):
    def run(video_name):
        data_gen = partial(data_generator.run, video_name=video_name)
        tf_dset = tf.data.Dataset.from_generator(
            data_gen, tf.float32, tf.TensorShape([batch_size, image_shape[0], image_shape[1], 1]),
        )
        return tf_dset

    return run


def _can_resolve_results(shuffled_results: ShuffledResults, score_threshold: float, video_name: str) -> bool:
    scores = shuffled_results.scores
    if np.all(np.isnan(scores)):
        logger.error(f"Calculated scores are all invalid, stopping analysis for {video_name}")
        return False

    if np.max(scores) < score_threshold:
        logger.error(
            f"There is not one frame where the error metric is above the threshold {score_threshold} "
            f"in the whole video {video_name}, stopping analysis. Maybe the model didn't train properly."
        )
        return False
    return True


class _Predictor(object):
    def __init__(self, results_scoring: ResultsScoring, keras_model):
        self.keras_model = keras_model
        self.results_scoring = results_scoring

    def __call__(
        self, num_frames: int, input_frames, scoring_data_manager: BaseScoringDataManager, features: Features,
    ) -> Tuple[OriginalResults, ShuffledResults]:
        # run all frames through the neural network to get a result theta without head/tail decision
        network_predictions = self.keras_model.predict(input_frames)[:num_frames]
        logger.info(f"Predicted {len(network_predictions)} frames")

        shuffled_results = ShuffledResults(random_theta=network_predictions)

        original_results = OriginalResults(
            theta=skeletons_to_angles(features.skeletons, theta_dims=network_predictions.shape[1]),
            skeletons=features.skeletons,
            scores=None,
        )

        # calculate image similarity for each frame, for the two solutions
        self.results_scoring(results=shuffled_results, scoring_data_manager=scoring_data_manager)

        avg_score = np.max(shuffled_results.scores, axis=1).mean()
        logger.info(f"Calculated image similarity, average: {avg_score:.4f}")

        resample_results(shuffled_results, features.timestamp)
        resample_results(original_results, features.timestamp)

        return original_results, shuffled_results


def _apply_resize_factor(results: BaseResults, resize_factor: float):
    results.skeletons /= resize_factor


def _parse_arguments(dataset_path: str, kwargs: dict):
    if kwargs.get("work_dir") is None:
        kwargs["work_dir"] = default_paths.WORK_DIR
    if kwargs.get("num_process") is None:
        kwargs["num_process"] = os.cpu_count()
    if kwargs.get("temp_dir") is None:
        kwargs["temp_dir"] = tempfile.gettempdir()
    if kwargs.get("batch_size") is None:
        kwargs["batch_size"] = 512
    if kwargs.get("score_threshold") is None:
        kwargs["score_threshold"] = 0.7
    if kwargs.get("video_names") is None:
        kwargs["video_names"] = None
    if kwargs.get("random_seed") is None:
        kwargs["random_seed"] = None

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


def predict(dataset_path: str, **kwargs):
    """
    Use a trained model to predict the centerlines of worm for videos in a dataset

    :param dataset_path: Root path of the dataset containing videos of worm
    """
    args = _parse_arguments(dataset_path, kwargs)

    mp.set_start_method("spawn", force=True)

    if args.random_seed is not None:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        tf.random.set_seed(args.random_seed)

    results_root_dir = os.path.join(args.experiment_dir, default_paths.RESULTS_DIR)
    os.makedirs(results_root_dir, exist_ok=True)

    config = load_config(args.config)

    dataset = load_dataset(
        dataset_loader=config.dataset_loader,
        dataset_path=dataset_path,
        selected_video_names=args.video_names,
        resize_options=ResizeOptions(resize_factor=config.resize_factor),
    )

    keras_model = tf.keras.models.load_model(args.model_path, compile=False)

    results_saver = ResultsSaver(
        temp_dir=args.temp_dir, results_root_dir=results_root_dir, results_filename=RESULTS_FILENAME
    )

    tf_dataset_maker = _make_tf_dataset(
        data_generator=PredictDataGenerator(
            dataset=dataset,
            num_process=args.num_process,
            temp_dir=args.temp_dir,
            image_shape=config.image_shape,
            batch_size=args.batch_size,
        ),
        batch_size=args.batch_size,
        image_shape=config.image_shape,
    )

    results_scoring = ResultsScoring(
        frame_preprocessing=dataset.frame_preprocessing,
        num_process=args.num_process,
        temp_dir=args.temp_dir,
        image_shape=config.image_shape,
    )
    predictor = _Predictor(results_scoring=results_scoring, keras_model=keras_model)

    for video_name in dataset.video_names:
        logger.info(f'Processing video: "{video_name}"')
        features = dataset.features_dataset[video_name]

        template_indexes = features.labelled_indexes
        if len(template_indexes) == 0:
            logger.error(
                f"Can't calculate image metric, there is no labelled frame in the video to use as a template, "
                f"stopping analysis for {video_name}."
            )
            continue

        original_results, shuffled_results = predictor(
            input_frames=tf_dataset_maker(video_name),
            num_frames=dataset.num_frames(video_name),
            features=features,
            scoring_data_manager=ScoringDataManager(
                video_name=video_name, frames_dataset=dataset.frames_dataset, features=features,
            ),
        )

        results = {"original": original_results, "unaligned": shuffled_results}
        if _can_resolve_results(shuffled_results, video_name=video_name, score_threshold=args.score_threshold,):
            final_results = resolve_head_tail(
                shuffled_results=shuffled_results,
                original_results=original_results,
                frame_rate=features.frame_rate,
                score_threshold=args.score_threshold,
            )
            results["resolved"] = final_results
            _apply_resize_factor(results["resolved"], config.resize_factor)

        _apply_resize_factor(results["unaligned"], config.resize_factor)

        results_saver.save(results=results, video_name=video_name)

    # cleanup
    shutil.rmtree(args.temp_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser()

    # model infos
    parser.add_argument(
        "--model_path", type=str, help="Load model from this path, or use best model from work_dir.",
    )
    parser.add_argument("--batch_size", type=int)

    # inputs
    parser.add_argument("dataset_path", type=str)
    parser.add_argument(
        "--video_names",
        type=str,
        nargs="+",
        help="Only analyze a subset of videos. If not set, will analyze all videos in dataset_path.",
    )
    add_config_argument(parser)
    parser.add_argument("--temp_dir", type=str, help="Where to store temporary intermediate results")
    parser.add_argument("--work_dir", type=str, help="Root folder for all experiments")
    # multiprocessing params
    parser.add_argument("--num_process", type=int, help="How many worker processes")
    # parameters of results processing
    parser.add_argument(
        "--score_threshold",
        type=float,
        help="Image metric score threshold : discard results scoring lower than this value."
        " Fine tune this value using the script calibrate_dataset.py",
    )
    parser.add_argument("--random_seed", type=int, help="Optional random seed for deterministic results")
    args = parser.parse_args()

    predict(**vars(args))


if __name__ == "__main__":
    main()
