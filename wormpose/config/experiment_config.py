import json
import os

from typing import Tuple, List


def add_config_argument(parser):
    """
    For command line arguments, add the option to pass the path of the configuration file

    :param parser: Argparse parser
    """
    parser.add_argument(
        "--config",
        type=str,
        help="Path of the configuration file."
        " The file is created when generating the training dataset."
        " If not set, will look for it in the default location: at"
        " {work_dir}/{dataset_name}/config.json",
    )


class ExperimentConfig(object):
    """
    Data container for the experiment config, created when generating training data
    """

    def __init__(
        self,
        num_train_samples: int = None,
        num_eval_samples: int = None,
        image_shape: Tuple[int, int] = None,
        dataset_loader: str = None,
        theta_dimensions: int = None,
        resize_factor: float = None,
        video_names: List[str] = None,
    ):
        self.num_train_samples = num_train_samples
        self.num_eval_samples = num_eval_samples
        self.image_shape = image_shape
        self.dataset_loader = dataset_loader
        self.theta_dimensions = theta_dimensions
        self.resize_factor = resize_factor
        self.video_names = video_names


def save_config(experiment_config: ExperimentConfig, config_filepath: str):
    """
    Save the experiment config to a json file

    :param experiment_config: config object to save
    :param config_filepath: path where to write the config json file
    """
    with open(config_filepath, "w") as f:
        json.dump(experiment_config, f, indent=4, default=lambda x: x.__dict__)


def load_config(config_filepath: str) -> ExperimentConfig:
    """
    Load the experiment config from a json file

    :param config_filepath: path of the config json file to load
    :return: loaded config object
    """
    if not os.path.isfile(config_filepath):
        raise FileNotFoundError(f"Configuration file not found at path: '{config_filepath}'.")

    with open(config_filepath, "r") as f:
        return ExperimentConfig(**json.load(f))
