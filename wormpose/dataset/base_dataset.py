"""
This module contains the WormPose API: abstract classes to subclass in order to add a custom dataset
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseFramePreprocessing(ABC):
    """
    Specific image processing logic to isolate the worm in the dataset images
    This object must be pickable (no inner functions for example)
    """

    @abstractmethod
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Segment the worm object of interest in the image: returns a mask image of the same shape as frame,
        where the pixels belong to the worm object of interest are 1, and all the others are 0
        Also calculates the average value of the background pixels.

        :param frame: image to process
        :return: Segmentation mask image , background color
        """
        pass


class BaseFramesDataset(ABC):
    """
    Specific code to the dataset to access the frames only.
    A dataset is divided into several "videos". Each video contain a list of images or "frames".
    """

    @abstractmethod
    def video_names(self) -> List[str]:
        """
        A dataset is composed of several videos

        :return: A list of unique ids (string) identifying a video in the dataset
        """
        pass

    @abstractmethod
    def open(self, video_name: str):
        """
        The frames of the dataset are accessed trough a context manager object, in this way we have the option of not
        entirely loading a big image array in memory if possible

        :param video_name: One video unique id (should be one value of video_names())
        :return: A context manager object that can be used with the "with" python statement giving access to the
            frames (array of images) of the dataset
            Example use : frames = open("video0")
        """
        pass


class BaseFeaturesDataset(ABC):
    """
     Specific code to the dataset to access the features for each video of the dataset
     """

    @abstractmethod
    def get_features(self, video_name: str) -> dict:
        """
        Returns a dictionary of features

        :return: dictionary with keys: skeletons, head_width, midbody_width, tail_width, frame_rate, ventral_side, timestamp
            WHERE
            skeletons: Coordinates x y of the centerline for each frame in pixel coordinates,
            a numpy floating point array of shape (N number of frames, J number of joints, 2)
            The quality of the synthetic images will start degrading when J < 50, consider interpolating if less joints
            head_width: numpy floating point array of shape N
            midbody_width: numpy floating point array of shape N
            tail_width: numpy floating point array of shape N
            frame_rate: One float number for the frame rate of the video.
            ventral_side: Optional One string value for the entire video. \'clockwise\' or \'anticlockwise\'. If None, defaults to anticlockwise
            timestamp: Optional Timestamp of each frame, a numpy array of shape (N number of frames). If None, will consider each frame to be equidistant in time
        """
        pass


class BaseResultsExporter(ABC):
    """
    Optional Results Exporter
    """

    @abstractmethod
    def export(self, video_name: str, **kwargs):
        pass
