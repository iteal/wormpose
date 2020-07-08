#!/usr/bin/env python

"""
Post-processes WormPose results by interpolating over missing frames and smoothing
"""

import glob
import logging
import os
import tempfile
from argparse import Namespace
from typing import Sequence

import h5py
import numpy as np
import numpy.ma as ma
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

from wormpose.commands import _log_parameters
from wormpose.commands.utils.results_saver import ResultsSaver
from wormpose.commands.utils.time_sampling import resample_results
from wormpose.config import default_paths
from wormpose.config.default_paths import RESULTS_FILENAME, POSTPROCESSED_RESULTS_FILENAME, CONFIG_FILENAME
from wormpose.config.experiment_config import load_config, add_config_argument
from wormpose.dataset.loader import get_dataset_name, Dataset
from wormpose.dataset.loader import load_dataset
from wormpose.images.scoring import ResultsScoring, ScoringDataManager
from wormpose.pose.eigenworms import load_eigenworms_matrix
from wormpose.pose.results_datatypes import BaseResults, OriginalResults

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _find_runs_boundaries(arr: np.ndarray) -> np.ndarray:
    """
    :return: list of pair of indexes [start, end] of the segments (runs of consecutive True values) boundaries
    """
    padded_arr = np.concatenate([[False], arr, [False]])
    return np.where(np.abs(np.diff(padded_arr)))[0].reshape(-1, 2)


def _get_valid_segments(is_valid_series: np.ndarray, max_gap_size: int, min_segment_size: int) -> Sequence:
    """
    :return: list of pair of indexes [start, end] of the valid segments in the data:
        They can contain small gaps <= max_gap_size but they need to be surrounded by valid data
        of length >= min_segment_size
    """

    # identify segments (consecutive runs of valid frames)
    segments_boundaries = _find_runs_boundaries(is_valid_series)

    # find the big segments of length greater or equal than min_segment_size
    big_segments_boundaries = [[start, end] for start, end in segments_boundaries if end - start >= min_segment_size]
    big_segments = np.full_like(is_valid_series, False)
    for gap_start, gap_end in big_segments_boundaries:
        big_segments[gap_start:gap_end] = True

    # identify gaps (consecutive runs of invalid frames between the big segments)
    gaps_boundaries = _find_runs_boundaries(~big_segments)

    # only keep the big gaps of length greater than max_gap_size
    big_gaps_boundaries = [[start, end] for start, end in gaps_boundaries if end - start > max_gap_size]
    big_gaps = np.full_like(is_valid_series, False)
    for gap_start, gap_end in big_gaps_boundaries:
        big_gaps[gap_start:gap_end] = True

    # the final valid segments are in-between the big gaps
    # they can contain small gaps but there are at least min_segment_size before and after the small gaps
    valid_segments_boundaries = _find_runs_boundaries(~big_gaps)

    return valid_segments_boundaries


class _SplineInterpolation(object):
    def __init__(self, frames_around: int = 3, spline_degree: int = 3):

        self.spline_degree = spline_degree
        self.frames_around = frames_around
        self.slope = np.linspace(0, 0.5, frames_around + 1)

    def interpolate_tseries(
        self, tseries: np.ndarray, segments_boundaries: Sequence, std_fraction: float
    ) -> np.ndarray:

        weight = 1 / (std_fraction * np.nanstd(tseries))

        tseries[~np.isnan(tseries)] = np.unwrap(tseries[~np.isnan(tseries)])
        new_tseries = np.full_like(tseries, np.nan)

        for t0, tf in segments_boundaries:
            new_tseries[t0:tf] = self._interpolate_segment(tseries[t0:tf], weight)

        return new_tseries

    def _interpolate_segment(self, tseries: np.ndarray, weight: float) -> np.ndarray:
        new_tseries = np.copy(tseries)

        nan_y = np.isnan(new_tseries)
        indices_nan = np.any(nan_y, axis=1)
        series_len = len(new_tseries)
        x = np.arange(series_len)
        new_tseries[nan_y] = 0.0

        w = self.build_weights(indices_nan, series_len, weight)

        # perform spline interpolation separately for each dimension
        for dim in range(new_tseries.shape[1]):
            y = new_tseries[:, dim]
            spl = UnivariateSpline(x, y, w=w, k=self.spline_degree, s=len(x))
            new_x = x
            new_weighted_y = spl(new_x)
            new_tseries[:, dim] = new_weighted_y

        return new_tseries

    def build_weights(self, indices_nan: np.ndarray, series_len: int, weight: float) -> np.ndarray:
        # setup weights: lower the weights closer to the edges
        w = np.full(series_len, weight)
        where_nan = np.where(indices_nan)[0]
        if len(where_nan) == 0:
            return w

        for idx in range(series_len):
            closest_nan_distance = np.min(np.abs(where_nan - idx))
            if closest_nan_distance <= self.frames_around:
                w[idx] = self.slope[closest_nan_distance] * weight

        return w


def _smooth_tseries(
    tseries: np.ndarray, smoothing_window_length: int, poly_order: int, segments_boundaries: Sequence,
) -> np.ndarray:
    if smoothing_window_length % 2 == 0:
        smoothing_window_length += 1  # need a odd number for smoothing_window_length

    new_tseries = np.full_like(tseries, np.nan)

    for t0, tf in segments_boundaries:
        if tf - t0 < smoothing_window_length or tf - t0 < poly_order:
            continue
        new_tseries[t0:tf] = savgol_filter(
            tseries[t0:tf], axis=0, window_length=smoothing_window_length, polyorder=poly_order,
        )

    return new_tseries


def _dorsal_ventral_flip_theta(theta: np.ndarray) -> np.ndarray:
    return 2 * np.pi - theta


def _thetas_to_modes(thetas: np.ndarray, eigenworms_matrix: np.ndarray) -> np.ndarray:
    return (thetas.T - thetas.mean(axis=1)).T.dot(eigenworms_matrix)


def _unwrap_ma(x: ma.MaskedArray):
    idx = ma.array(np.arange(0, x.shape[0]), mask=x.mask)
    idxc = idx.compressed()
    xc = x.compressed()
    dd = np.diff(xc)
    ddmod = np.mod(dd + np.pi, 2 * np.pi) - np.pi
    ddmod[(ddmod == -np.pi) & (dd > 0)] = np.pi
    phc_correct = ddmod - dd
    phc_correct[np.abs(dd) < np.pi] = 0
    ph_correct = np.zeros(x.shape)
    ph_correct[idxc[1:]] = phc_correct
    up = x + ph_correct.cumsum()
    return up


def _calculate_skeleton(theta: np.ndarray, args, dataset: Dataset, video_name: str) -> BaseResults:
    frames_timestamp = dataset.features_dataset[video_name].timestamp
    features = dataset.features_dataset[video_name]

    # resample time serie to have the same length as the number of frames
    theta_resampled = np.empty((dataset.num_frames(video_name),) + theta.shape[1:], dtype=theta.dtype)
    for cur_time, cur_theta in enumerate(theta):
        frame_index = np.where(frames_timestamp == cur_time)[0]
        theta_resampled[frame_index] = cur_theta

    results = BaseResults(theta=theta_resampled)
    ResultsScoring(
        frame_preprocessing=dataset.frame_preprocessing,
        num_process=args.num_process,
        temp_dir=args.temp_dir,
        image_shape=dataset.image_shape,
    )(
        results=results,
        scoring_data_manager=ScoringDataManager(
            video_name=video_name, frames_dataset=dataset.frames_dataset, features=features,
        ),
    )
    resample_results(results, features.timestamp)

    return results


def _parse_arguments(dataset_path: str, kwargs: dict):
    if kwargs.get("work_dir") is None:
        kwargs["work_dir"] = default_paths.WORK_DIR
    if kwargs.get("max_gap_size") is None:
        kwargs["max_gap_size"] = 3
    if kwargs.get("min_segment_size") is None:
        kwargs["min_segment_size"] = 30
    if kwargs.get("smoothing_window") is None:
        kwargs["smoothing_window"] = 7
    if kwargs.get("poly_order") is None:
        kwargs["poly_order"] = 3
    if kwargs.get("std_fraction") is None:
        kwargs["std_fraction"] = 0.01
    if kwargs.get("eigenworms_matrix_path") is None:
        kwargs["eigenworms_matrix_path"] = None
    if kwargs.get("num_process") is None:
        kwargs["num_process"] = os.cpu_count()
    if kwargs.get("temp_dir") is None:
        kwargs["temp_dir"] = tempfile.gettempdir()
    kwargs["temp_dir"] = tempfile.mkdtemp(dir=kwargs["temp_dir"])

    dataset_name = get_dataset_name(dataset_path)
    kwargs["experiment_dir"] = os.path.join(kwargs["work_dir"], dataset_name)

    if kwargs.get("config") is None:
        kwargs["config"] = os.path.join(kwargs["experiment_dir"], CONFIG_FILENAME)

    _log_parameters(logger.info, {"dataset_path": dataset_path})
    _log_parameters(logger.info, kwargs)

    return Namespace(**kwargs)


def post_process(dataset_path: str, **kwargs):
    """
    Process the raw network results with interpolation and smoothing

    :param dataset_path: Root path of the dataset containing videos of worm
    """
    args = _parse_arguments(dataset_path, kwargs)

    results_root_dir = os.path.join(args.experiment_dir, default_paths.RESULTS_DIR)

    eigenworms_matrix = load_eigenworms_matrix(args.eigenworms_matrix_path)

    config = load_config(args.config)

    dataset = load_dataset(config.dataset_loader, dataset_path)

    spline_interpolation = _SplineInterpolation()

    results_files = list(sorted(glob.glob(os.path.join(results_root_dir, "*", RESULTS_FILENAME))))
    if len(results_files) == 0:
        raise FileNotFoundError("No results file to analyze was found")

    for results_file in results_files:
        video_name = os.path.basename(os.path.dirname(results_file))

        with h5py.File(results_file, "r") as results_f:

            try:
                results_raw = BaseResults(
                    theta=results_f["resolved"]["theta"][:],
                    skeletons=results_f["resolved"]["skeletons"][:],
                    scores=results_f["resolved"]["scores"][:],
                )
            except Exception:
                logger.error(f"Couldn't read results in file {results_file}.")
                continue

            results_orig = OriginalResults(
                theta=results_f["original"]["theta"][:], skeletons=results_f["original"]["skeletons"][:]
            )

            features = dataset.features_dataset[video_name]

            missing_values = np.any(np.isnan(results_raw.theta), axis=1)
            if missing_values.sum() == len(results_raw.theta):
                logger.warning(f"No valid result was found, stopping postprocessing for {video_name}")
                continue

            segments_boundaries = _get_valid_segments(
                is_valid_series=~missing_values, max_gap_size=args.max_gap_size, min_segment_size=args.min_segment_size,
            )
            # interpolate and smooth in angles space
            thetas_interp = spline_interpolation.interpolate_tseries(
                results_raw.theta, segments_boundaries, args.std_fraction
            )
            results_interp = _calculate_skeleton(thetas_interp, args, dataset, video_name)

            thetas_smooth = _smooth_tseries(thetas_interp, args.smoothing_window, args.poly_order, segments_boundaries,)
            results_smooth = _calculate_skeleton(thetas_smooth, args, dataset, video_name)

            flipped = False

            if features.ventral_side == "clockwise":
                results_orig.theta = _dorsal_ventral_flip_theta(results_orig.theta)
                results_raw.theta = _dorsal_ventral_flip_theta(results_raw.theta)
                results_interp.theta = _dorsal_ventral_flip_theta(results_interp.theta)
                results_smooth.theta = _dorsal_ventral_flip_theta(results_smooth.theta)
                flipped = True

            if eigenworms_matrix is not None:
                setattr(results_orig, "modes", _thetas_to_modes(results_orig.theta, eigenworms_matrix))
                setattr(results_raw, "modes", _thetas_to_modes(results_raw.theta, eigenworms_matrix))
                setattr(results_interp, "modes", _thetas_to_modes(results_interp.theta, eigenworms_matrix))
                setattr(results_smooth, "modes", _thetas_to_modes(results_smooth.theta, eigenworms_matrix))

        # save results
        results_saver = ResultsSaver(
            temp_dir=args.temp_dir, results_root_dir=results_root_dir, results_filename=POSTPROCESSED_RESULTS_FILENAME
        )

        metadata = {
            "max_gap_size": args.max_gap_size,
            "min_segment_size": args.min_segment_size,
            "smoothing_window": args.smoothing_window,
            "poly_order": args.poly_order,
            "std_fraction": args.std_fraction,
            "dorsal_ventral_flip": flipped,
        }

        results_saver.save(
            results={"orig": results_orig, "raw": results_raw, "interp": results_interp, "smooth": results_smooth},
            metadata=metadata,
            video_name=video_name,
        )
        logger.info(f"Post-processed worm: {video_name} {'(flipped dorsal-ventral)' if flipped else ''}")


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_path", type=str)
    parser.add_argument(
        "--eigenworms_matrix_path", help="Path to optional eigenworms matrix to also save results in eigenworm modes",
    )
    parser.add_argument("--work_dir", type=str, help="Root folder for all experiments")
    add_config_argument(parser)
    parser.add_argument(
        "--max_gap_size",
        type=float,
        help="Interpolate over missing values (gaps), as long"
        "as the consecutive length of the missing values is less than max_gap_size (frames)",
    )
    parser.add_argument(
        "--min_segment_size",
        type=float,
        help="Only segments of valid values of length greater than min_segment_size (frames)"
        "will be interpolated and smoothed",
    )
    parser.add_argument(
        "--std_fraction",
        type=int,
        help="The higher the guessed noise to signal ratio is, the smoother the interpolation will be",
    )
    parser.add_argument("--smoothing_window", type=int, help="smoothing window in frames")
    parser.add_argument("--poly_order", type=int, help="polynomial order in smoothing")
    parser.add_argument("--temp_dir", type=str, help="Where to store temporary intermediate results")
    parser.add_argument("--num_process", type=int, help="How many worker processes")
    args = parser.parse_args()

    post_process(**vars(args))


if __name__ == "__main__":
    main()
