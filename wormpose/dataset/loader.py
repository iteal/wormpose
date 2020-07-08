"""
The Dataset loader: instantiates the FramesDataset, FeaturesDataset, FramePreprocessing, ResultsExporter (optional).
Also handles the resizing options.
"""

import os
from typing import List, Optional, Tuple

import pkg_resources

from wormpose.dataset.base_dataset import (
    BaseFramesDataset,
    BaseFramePreprocessing,
    BaseResultsExporter,
)
from wormpose.dataset.features import Features, calculate_crop_window_size, FeaturesDict
from wormpose.dataset.loaders.resizer import (
    frames_dataset_resizer,
    features_dataset_resizer,
    ResizeOptions,
)


class Dataset(object):
    def __init__(
        self,
        video_names: List[str],
        frames_dataset: BaseFramesDataset,
        features_dataset: FeaturesDict,
        frame_preprocessing: BaseFramePreprocessing,
        image_shape: Tuple[int, int],
        results_exporter: BaseResultsExporter,
    ):
        self.video_names = video_names
        self.frames_dataset = frames_dataset
        self.features_dataset = features_dataset
        self.frame_preprocessing = frame_preprocessing
        self.image_shape = image_shape
        self.results_exporter = results_exporter

    def num_frames(self, video_name):
        with self.frames_dataset.open(video_name) as frames:
            return len(frames)


def get_dataset_name(dataset_path: str) -> str:
    """
    Each dataset gets assigned a name: the root folder of the dataset

    :param dataset_path: Full path of the dataset
    :return: Name identifier for the dataset, simply use the basename of the path
        Each different dataset must have a unique basename in order to process several at once
    """
    return os.path.basename(os.path.normpath(dataset_path))


class _DummyResultsExporter(BaseResultsExporter):
    """
    Does nothing
    """

    def export(self, video_name: str, **kwargs):
        pass


def load_dataset(
    dataset_loader: str,
    dataset_path: str,
    selected_video_names: Optional[List[str]] = None,
    resize_options: ResizeOptions = None,
) -> Dataset:
    for entry_point in pkg_resources.iter_entry_points("worm_dataset_loaders"):
        if entry_point.name == dataset_loader:
            module = entry_point.load()

            frames_dataset_class = module.FramesDataset
            features_dataset_class = module.FeaturesDataset
            frame_preprocessing_class = module.FramePreprocessing

            frames_dataset, features_dataset, video_names = _load_dataset(
                frames_dataset_class, features_dataset_class, dataset_path, selected_video_names,
            )

            frame_preprocessing = frame_preprocessing_class()
            image_shape = calculate_crop_window_size(features_dataset)

            if resize_options is not None:
                resize_options.update_resize_factor(features_dataset)
                if resize_options.resize_factor != 1.0:
                    # reload frames and features dataset after resizing, also get new image_shape
                    frames_dataset_class = frames_dataset_resizer(
                        frames_dataset_class, resize_factor=resize_options.resize_factor
                    )
                    features_dataset_class = features_dataset_resizer(
                        features_dataset_class, resize_factor=resize_options.resize_factor,
                    )
                    frames_dataset, features_dataset, video_names = _load_dataset(
                        frames_dataset_class, features_dataset_class, dataset_path, selected_video_names,
                    )
                    image_shape = resize_options.get_image_shape(features_dataset)

            results_exporter = (
                module.ResultsExporter(dataset_path) if hasattr(module, "ResultsExporter") else _DummyResultsExporter()
            )

            return Dataset(
                video_names=video_names,
                features_dataset=features_dataset,
                frames_dataset=frames_dataset,
                frame_preprocessing=frame_preprocessing,
                image_shape=image_shape,
                results_exporter=results_exporter,
            )

    raise NotImplementedError(f"Dataset loader: '{dataset_loader}' not found in the package entry points.")


def _load_dataset(
    frames_dataset_class, features_dataset_class, dataset_path: str, selected_video_names,
):
    frames_dataset = frames_dataset_class(dataset_path)
    video_names = _resolve_video_names(frames_dataset, selected_video_names)

    raw_features_dataset = features_dataset_class(dataset_path, video_names)
    features_dataset = {}
    for video_name in video_names:
        features_dataset[video_name] = Features(raw_features_dataset.get_features(video_name))
    return frames_dataset, features_dataset, video_names


def _resolve_video_names(frames_dataset: BaseFramesDataset, selected_video_names: Optional[List[str]]):
    video_names = frames_dataset.video_names()

    if selected_video_names is not None:
        for video_name in selected_video_names:
            if video_name not in video_names:
                raise ValueError(f"Requested video '{video_name}' not found in dataset ({video_names})")
        video_names = selected_video_names

    _validate_video_names(video_names)

    return video_names


def _validate_video_names(video_names: List[str]):
    if len(video_names) == 0:
        raise ValueError(f"Video names list is empty, no video to analyze")
    if not all(isinstance(n, str) for n in video_names):
        raise NotImplementedError("Only video names as list of strings supported type.")
    if len(video_names) > len(set(video_names)):
        raise ValueError(f"Video names must be unique but duplicates were found: {video_names}.")
