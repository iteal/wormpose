"""
This module performs image scoring on the results (shuffled: two scores per frame or not)
It distributes the work to several processes
"""

import logging
import os
import pickle
from multiprocessing import Process, Manager
from typing import Tuple

import numpy as np

from wormpose import BaseFramePreprocessing
from wormpose.images.scoring.scoring_data_manager import BaseScoringDataManager
from wormpose.images.scoring.centerline_accuracy_check import CenterlineAccuracyCheck
from wormpose.pose.results_datatypes import BaseResults

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_CHUNK_SIZE = 100


class ResultsScoring(object):
    def __init__(
        self,
        frame_preprocessing: BaseFramePreprocessing,
        image_shape: Tuple[int, int],
        temp_dir: str,
        num_process: int,
    ):
        self.frame_preprocessing = frame_preprocessing
        self.image_shape = image_shape
        self.temp_dir = temp_dir
        self.num_process = num_process
        self.data_reading_queue = None

    def __call__(
        self, results: BaseResults, scoring_data_manager: BaseScoringDataManager,
    ):

        manager = Manager()
        self.data_reading_queue = manager.Queue()
        results_queue = manager.Queue()

        workers = [
            Process(
                target=_compare_pred_real,
                args=(
                    self.frame_preprocessing,
                    self.data_reading_queue,
                    results_queue,
                    self.image_shape,
                    self.temp_dir,
                ),
            )
            for _ in range(self.num_process)
        ]
        for w in workers:
            w.start()

        self._produce_data_chunks(
            results_theta=results.theta, scoring_data_manager=scoring_data_manager,
        )

        for w in workers:
            w.join()

        results_files = []
        while not results_queue.empty():
            results_files.append(results_queue.get())
        results_files = sorted(results_files)

        matching_scores = []
        matching_skeletons = []
        for result_filename in results_files:
            with open(result_filename, "rb") as f:
                results_scores, results_skel = pickle.load(f)
            matching_scores.append(results_scores)
            matching_skeletons.append(results_skel)
            os.remove(result_filename)

        results.scores = np.concatenate(matching_scores)
        results.skeletons = np.concatenate(matching_skeletons)

    def _produce_data_chunks(self, results_theta: np.ndarray, scoring_data_manager: BaseScoringDataManager):

        chunk_data = []
        chunk_frames = []
        chunk_index = 0
        chunk_template_frames = []

        with scoring_data_manager as scoring_data:

            for frame_index, thetas in enumerate(results_theta):

                (template_frame, template_skeleton, template_measurements, frame,) = scoring_data[frame_index]

                chunk_template_frames.append(template_frame)
                chunk_data.append((thetas, template_skeleton, template_measurements))
                chunk_frames.append(frame)

                if len(chunk_data) >= _CHUNK_SIZE or frame_index == len(results_theta) - 1:
                    self._save_chunk(
                        chunk_data=chunk_data,
                        chunk_index=chunk_index,
                        real_frames=chunk_frames,
                        template_frames=chunk_template_frames,
                    )
                    chunk_frames = []
                    chunk_index += 1
                    chunk_data = []
                    chunk_template_frames = []

        for _ in range(self.num_process):
            self.data_reading_queue.put(None)

    def _save_chunk(self, chunk_data, chunk_index, real_frames, template_frames):
        compare_data_filename = os.path.join(self.temp_dir, f"compare_data_{chunk_index}.pkl")
        with open(compare_data_filename, "wb") as f:
            pickle.dump((chunk_data, real_frames, template_frames), f)

        self.data_reading_queue.put((compare_data_filename, chunk_index))


def _compare_pred_real(
    frame_preprocessing: BaseFramePreprocessing, data_reading_queue, results_queue, image_shape, temp_dir: str,
):
    centerline_accuracy = CenterlineAccuracyCheck(frame_preprocessing=frame_preprocessing, image_shape=image_shape)
    while True:
        queue_data = data_reading_queue.get()
        if queue_data is None:
            break

        compare_data_filename, chunk_index = queue_data

        with open(compare_data_filename, "rb") as f:
            chunk_data, chunk_real_frames, chunk_template_frames = pickle.load(f)
        os.remove(compare_data_filename)

        all_scores = None
        all_skel = None
        for index, (data, real_frame_orig, template_frame) in enumerate(
            zip(chunk_data, chunk_real_frames, chunk_template_frames)
        ):
            thetas, template_skeleton, template_measurements = data

            # reshape thetas so that we can handle all cases (one centerline or two or more centerlines to score)
            if len(thetas.shape) == 1:
                thetas = np.reshape(thetas, (1,) + thetas.shape)

            if all_scores is None:
                all_scores = np.empty((len(chunk_data), thetas.shape[0]), dtype=float)
            if all_skel is None:
                all_skel = np.empty(
                    (len(chunk_data), thetas.shape[0],) + template_skeleton.shape, dtype=template_skeleton.dtype
                )

            for flip_index, cur_theta in enumerate(thetas):

                score, synth_skel = centerline_accuracy(
                    theta=cur_theta,
                    template_skeleton=template_skeleton,
                    template_frame=template_frame,
                    template_measurements=template_measurements,
                    real_frame_orig=real_frame_orig,
                )

                all_scores[index][flip_index] = score
                all_skel[index][flip_index] = synth_skel

        # for one centerline case
        all_scores = all_scores.squeeze()
        all_skel = all_skel.squeeze()

        out_filename = os.path.join(temp_dir, f"compare_results_{chunk_index:09d}.pkl")

        with open(out_filename, "wb") as f:
            pickle.dump((all_scores, all_skel), f)

        results_queue.put(out_filename)
