"""
This module exposes the gaussian mixture model of worm postures
from the resource file "postures_model.json.gz"
"""

import gzip
import json
import logging
import os
from typing import Generator

import numpy as np

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PosturesModel(object):
    def __init__(self):

        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir, "resources", "postures_model.json.gz",
        )

        with gzip.open(model_path, "rt") as f:
            shapes_model = json.load(f)

        self.means = np.array(shapes_model["means"])
        self.covariances = np.array(shapes_model["covariances"])
        self.weights = np.array(shapes_model["weights"])
        self.num_gaussians = self.means.shape[0]

    def generate(self) -> Generator:
        """
        Generates worm postures

        :return: a generator of numpy arrays of shape (N,) representing the centerline angles
        """
        while True:
            idx = np.random.choice(np.arange(self.num_gaussians), p=self.weights)
            random_global_angle = np.random.uniform(0, 2 * np.pi)

            samples = np.random.multivariate_normal(self.means[idx], self.covariances[idx], size=1)

            theta = samples[0] + random_global_angle

            yield theta
