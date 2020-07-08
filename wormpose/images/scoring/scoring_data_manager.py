from abc import ABC
from contextlib import AbstractContextManager
from typing import Sequence

import numpy as np

from wormpose import BaseFramesDataset


class BaseScoringDataManager(AbstractContextManager, Sequence, ABC):
    """
    Accessor for the data necessary to perform the scoring, including:
    - The image that needs to be scored
    - The image that will be used as a template for scoring, plus relevant infos: skeleton and measurements
    """

    pass


class ScoringDataManager(BaseScoringDataManager):
    """
    Implementation of the BaseScoringDataManager for scoring videos:
    it finds the closest template in time for a frame in a video
    """

    def __init__(self, frames_dataset: BaseFramesDataset, features, video_name: str):
        self.features = features
        self.video_name = video_name

        self._frames_dataset = frames_dataset
        self._frames_reader = None
        self._frames = None

    def __enter__(self):
        self._frames_reader = self._frames_dataset.open(self.video_name)
        self._frames = self._frames_reader.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._frames_reader.__exit__(exc_type, exc_value, traceback)
        self._frames_reader = None
        self._frames = None

    def __getitem__(self, frame_index: int):
        template_indexes = self.features.labelled_indexes
        # find the template image the most closely in time to the frame to score
        template_index = template_indexes[np.abs(template_indexes - frame_index).argmin()]
        return (
            self._frames[template_index],
            self.features.skeletons[template_index],
            self.features.measurements,
            self._frames[frame_index],
        )

    def __len__(self):
        return len(self._frames)
