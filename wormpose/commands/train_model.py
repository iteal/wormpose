#!/usr/bin/env python

"""
Trains the neural network on the training data, supports resuming training
"""

import glob
import logging
import multiprocessing as mp
import os
import random
from argparse import Namespace

import numpy as np
import tensorflow as tf

from wormpose.commands import _log_parameters
from wormpose.config import default_paths
from wormpose.config.default_paths import SYNTH_TRAIN_DATASET_NAMES, REAL_EVAL_DATASET_NAMES, CONFIG_FILENAME
from wormpose.config.experiment_config import load_config, add_config_argument
from wormpose.dataset.loader import get_dataset_name
from wormpose.machine_learning import model
from wormpose.machine_learning.best_models_saver import BestModels
from wormpose.machine_learning.loss import symmetric_angle_difference
from wormpose.machine_learning.tfrecord_file import get_tfrecord_dataset

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
tf.get_logger().setLevel(logging.INFO)


def _find_tfrecord_files(experiment_dir: str):
    training_data_dir = os.path.join(experiment_dir, default_paths.TRAINING_DATA_DIR)
    train_tfrecord_filenames = glob.glob(os.path.join(training_data_dir, SYNTH_TRAIN_DATASET_NAMES.format(index="*")))
    eval_tfrecord_filenames = glob.glob(os.path.join(training_data_dir, REAL_EVAL_DATASET_NAMES.format(index="*")))
    if len(train_tfrecord_filenames) == 0 or len(eval_tfrecord_filenames) == 0:
        raise FileNotFoundError("Training/Eval dataset not found.")
    return train_tfrecord_filenames, eval_tfrecord_filenames


def _parse_arguments(dataset_path: str, kwargs: dict):
    if kwargs.get("work_dir") is None:
        kwargs["work_dir"] = default_paths.WORK_DIR
    if kwargs.get("batch_size") is None:
        kwargs["batch_size"] = 128
    if kwargs.get("epochs") is None:
        kwargs["epochs"] = 100
    if kwargs.get("network_model") is None:
        kwargs["network_model"] = model.build_model
    if kwargs.get("optimizer") is None:
        kwargs["optimizer"] = "adam"
    if kwargs.get("loss") is None:
        kwargs["loss"] = symmetric_angle_difference
    if kwargs.get("random_seed") is None:
        kwargs["random_seed"] = None

    dataset_name = get_dataset_name(dataset_path)
    kwargs["experiment_dir"] = os.path.join(kwargs["work_dir"], dataset_name)

    if kwargs.get("config") is None:
        kwargs["config"] = os.path.join(kwargs["experiment_dir"], CONFIG_FILENAME)

    _log_parameters(logger.info, {"dataset_path": dataset_path})
    _log_parameters(logger.info, kwargs)

    return Namespace(**kwargs)


def train(dataset_path: str, **kwargs):
    """
    Train a neural network with the TFrecord files generated with the script generate_training_data
    Save the best model performing on evaluation data

    :param dataset_path: Root path of the dataset containing videos of worm
    """
    args = _parse_arguments(dataset_path, kwargs)

    mp.set_start_method("spawn", force=True)

    if args.random_seed is not None:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        tf.random.set_seed(args.random_seed)

    models_dir = os.path.join(args.experiment_dir, default_paths.MODELS_DIRS)
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    train_tfrecord_filenames, eval_tfrecord_filenames = _find_tfrecord_files(args.experiment_dir)

    config = load_config(args.config)
    if config.num_eval_samples < args.batch_size or config.num_train_samples < args.batch_size:
        raise ValueError("The number of samples in the train and eval datasets must be higher than the batch size.")

    train_dataset = get_tfrecord_dataset(
        filenames=train_tfrecord_filenames,
        image_shape=config.image_shape,
        batch_size=args.batch_size,
        theta_dims=config.theta_dimensions,
        is_train=True,
    )
    validation_dataset = get_tfrecord_dataset(
        filenames=eval_tfrecord_filenames,
        image_shape=config.image_shape,
        batch_size=args.batch_size,
        theta_dims=config.theta_dimensions,
        is_train=False,
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(args.experiment_dir, "tensorboard_log"), histogram_freq=1
    )
    best_models_callback = BestModels(models_dir)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        best_models_callback.models_name_pattern,
        save_best_only=False,
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
    )

    keras_model = args.network_model(input_shape=config.image_shape, out_dim=config.theta_dimensions)
    last_model_path = best_models_callback.last_model_path
    if os.path.isfile(last_model_path):
        keras_model = tf.keras.models.load_model(last_model_path, compile=False)

    keras_model.compile(optimizer=args.optimizer, loss=args.loss)
    keras_model.fit(
        train_dataset,
        epochs=args.epochs,
        steps_per_epoch=config.num_train_samples // args.batch_size,
        shuffle=False,
        initial_epoch=best_models_callback.epoch,
        validation_data=validation_dataset,
        validation_steps=config.num_eval_samples // args.batch_size,
        callbacks=[tensorboard_callback, checkpoint_callback, best_models_callback],
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_path", type=str)

    add_config_argument(parser)
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--epochs", type=int, help="How many epochs to train the network")
    parser.add_argument("--work_dir", type=str, help="Root folder for all experiments")
    parser.add_argument("--optimizer", type=str, help="Which optimizer for training, 'adam' by default.")
    parser.add_argument("--random_seed", type=int, help="Optional random seed for deterministic results")

    args = parser.parse_args()

    train(**vars(args))


if __name__ == "__main__":
    main()
