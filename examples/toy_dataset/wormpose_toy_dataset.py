"""
Example of a custom toy dataset
Install with the associated setup.py : then you can use "toy" as a dataset_loader for all wormpose commands
"""
import os
import pickle
from typing import Tuple

import cv2
import numpy as np

from wormpose import BaseFramePreprocessing, BaseFramesDataset, BaseFeaturesDataset

IM_SIZE = 100
MAX_BACKGROUND_COLOR = 30
WORM_BODY_COLOR_HEAD = 255
WORM_BODY_COLOR_TAIL = 100


def generate_toy_features(dataset_path: str, num_frames: int = 500, num_joints: int = 50):
    """
    Make up features for a toy video: the skeleton is a arc ellipse that rotates in time and the thickness is constant
    """

    skeletons = []
    head_width = []
    midbody_width = []
    tail_width = []

    init_angle = np.arange(0, num_frames)

    for i in range(num_frames):
        skel = []
        center = IM_SIZE // 2, IM_SIZE // 2
        worm_thickness = IM_SIZE // 20
        start_angle = 0
        end_angle = 180

        axes = IM_SIZE // 4, IM_SIZE // 3
        for theta in np.linspace(init_angle[i] + start_angle, init_angle[i] + end_angle, num_joints):
            x = int(center[0] + axes[0] * np.cos(np.deg2rad(theta)))
            y = int(center[1] - axes[1] * np.sin(np.deg2rad(theta)))
            skel.append((x, y))

        # same thickness everywhere along the body for this toy worm
        head_width.append(worm_thickness)
        midbody_width.append(worm_thickness)
        tail_width.append(worm_thickness)
        skeletons.append(skel)

    skeletons = np.array(skeletons, np.float32)
    head_width = np.array(head_width, np.float32)
    midbody_width = np.array(midbody_width, np.float32)
    tail_width = np.array(tail_width, np.float32)

    features = {
        "toy_video_0": {
            "skeletons": skeletons,
            "head_width": head_width,
            "midbody_width": midbody_width,
            "tail_width": tail_width,
            "frame_rate": 30,
        }
    }

    with open(dataset_path, "wb") as f:
        pickle.dump(features, f)


def draw_worm(frame, skel, width):
    body_color = np.linspace(WORM_BODY_COLOR_HEAD, WORM_BODY_COLOR_TAIL, len(skel) - 1)
    for i, (pt1, pt2) in enumerate(zip(skel[1:], skel[:-1])):
        cv2.line(
            frame, pt1=tuple(pt1.astype(int)), pt2=tuple(pt2.astype(int)), color=body_color[i], thickness=int(width)
        )


class FramesDataset(BaseFramesDataset):
    class FramesReader(object):
        """
        Create all toy frames images at init and keep in memory
        """

        def __init__(self, video_features):
            skeletons = video_features["skeletons"]
            midbody_width = video_features["midbody_width"]
            num_frames = len(skeletons)

            # we create a list of images with random background values between 0 to MAX_BACKGROUND_COLOR
            self.frames = np.random.random_integers(
                0, MAX_BACKGROUND_COLOR, size=(num_frames, IM_SIZE, IM_SIZE)
            ).astype(np.uint8)

            # we draw the worm body on each image
            for index in range(num_frames):
                draw_worm(self.frames[index], skeletons[index], midbody_width[index])

        def __enter__(self):
            """
            Here we just return the full frames in memory
            but for bigger datasets we can return a H5 dataset and close it in exit
            """
            return self.frames

        def __exit__(self, exc_type, exc_value, traceback):
            """
            No need to close resources here, we just keep all frames in memory
            """
            pass

    def __init__(self, dataset_path: str):
        """
        We create the toy features here and save to a file, FeaturesDataset will load them too
        """

        if not os.path.exists(dataset_path):
            generate_toy_features(dataset_path)

        with open(dataset_path, "rb") as f:
            self.features = pickle.load(f)

    def video_names(self):
        return list(self.features.keys())

    def open(self, video_name: str):
        return self.FramesReader(self.features[video_name])


class FeaturesDataset(BaseFeaturesDataset):
    def __init__(self, dataset_path: str, video_names):
        """
        We just load the features file here here because our toy dataset is initialized with generate_toy_features earlier
        """
        with open(dataset_path, "rb") as f:
            self.features = pickle.load(f)

    def get_features(self, video_name: str):
        """
        Return toy features
        """
        return self.features[video_name]


class FramePreprocessing(BaseFramePreprocessing):
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Calculates the foreground mask as all pixels that have a higher value than BACKGROUND_COLOR
        Also calculates the background color as the average of the background pixels
        """
        foreground_mask = frame > MAX_BACKGROUND_COLOR

        background_mask = ~foreground_mask
        background_color = int(np.mean(frame[background_mask]))

        return foreground_mask.astype(np.uint8), background_color
