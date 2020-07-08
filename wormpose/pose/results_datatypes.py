"""
This module contains data structures representing tracking results
They all contain theta, scores and skeletons but we use several subclasses to specify
the current step of the tracking pipeline
"""
from typing import Optional

import numpy as np

from wormpose.pose.centerline import flip_theta_series


class BaseResults(object):
    def __init__(
        self,
        theta: Optional[np.ndarray] = None,
        skeletons: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
    ):
        self.theta = theta
        self.scores = scores
        self.skeletons = skeletons

    def __len__(self):
        return len(self.theta)


class OriginalResults(BaseResults):
    pass


class ShuffledResults(BaseResults):
    def __init__(self, random_theta: np.ndarray):
        shuffled_theta = np.stack([random_theta, flip_theta_series(random_theta)], axis=1)
        super().__init__(theta=shuffled_theta, scores=None, skeletons=None)
