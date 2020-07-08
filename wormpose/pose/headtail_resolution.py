"""
This module contains the logic to resolve the head-tail orientation of a predicted video time series.
"""

import logging

import numpy as np
import numpy.ma as ma

from wormpose.pose.distance_metrics import angle_distance, skeleton_distance
from wormpose.pose.results_datatypes import (
    BaseResults,
    ShuffledResults,
    OriginalResults,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# threshold to compare neighbor frames theta, to be considered continuous and belong to the same segment
CONTINUOUS_ANGLES_DIST_THRESHOLD = np.deg2rad(30)

# we consider frames to be part of the same segment if they are maximum this amount of seconds apart
# (and satisfy the distance threshold)
CONTINOUS_SEGMENT_TIME_WINDOW_SEC = 0.2

# discard too small segments less than this amount of seconds
MIN_SEGMENT_SIZE_SEC = 0.2

# don't align isolated segments that are more than this amount of seconds apart from aligned segments
MAXIMUM_GAP_ALLOWED_WITH_ADJACENT_SEGMENT_SEC = 1


def _init_partitioned_series(shuffled_series: np.ndarray):
    return ma.masked_all_like(shuffled_series)


def _set_partition(partitioned_series, shuffled_series, frame_index: int, partition: int):
    partitioned_series[frame_index][0] = shuffled_series[frame_index, partition]
    partitioned_series[frame_index][1] = shuffled_series[frame_index, 1 - partition]


class _PartitionedResults(BaseResults):
    def __init__(self, shuffled_results: ShuffledResults):

        self.cur_partition = -1
        self.partitions = ma.masked_all((len(shuffled_results),), dtype=int)
        self._shuffled_results = shuffled_results

        theta = _init_partitioned_series(shuffled_results.theta)
        skeletons = _init_partitioned_series(shuffled_results.skeletons)
        scores = _init_partitioned_series(shuffled_results.scores)
        super().__init__(theta=theta, skeletons=skeletons, scores=scores)

    def mask(self, indices):
        self.theta.mask[indices] = True
        self.skeletons.mask[indices] = True
        self.scores.mask[indices] = True
        self.partitions.mask[indices] = True

    def set_partition(self, frame_index: int, partition: int, new_partition: bool = False):
        if new_partition:
            self.cur_partition += 1

        _set_partition(self.theta, self._shuffled_results.theta, frame_index, partition)
        _set_partition(self.skeletons, self._shuffled_results.skeletons, frame_index, partition)
        _set_partition(self.scores, self._shuffled_results.scores, frame_index, partition)
        self.partitions[frame_index] = self.cur_partition

    def _get_partition_indices(self, partition_index: int):
        return np.where(self.partitions == partition_index)[0]

    def get_segments(self):
        all_partitions_indexes = np.unique(self.partitions.filled(-1))
        return [
            self._get_partition_indices(partition_index)
            for partition_index in all_partitions_indexes
            if partition_index >= 0
        ]


class _ResolvedResults(BaseResults):
    def __init__(self, partitioned_results: _PartitionedResults):
        self._partitioned_results = partitioned_results
        theta = _init_unified_series(partitioned_results.theta)
        skeletons = _init_unified_series(partitioned_results.skeletons)
        scores = _init_unified_series(partitioned_results.scores)
        super().__init__(theta=theta, skeletons=skeletons, scores=scores)

    def resolve(self, segment, segment_alignment):
        self.scores[segment] = self._partitioned_results.scores[segment][:, segment_alignment]
        self.skeletons[segment] = self._partitioned_results.skeletons[segment][:, segment_alignment]
        self.theta[segment] = self._partitioned_results.theta[segment][:, segment_alignment]

    def mask(self, indices):
        self.theta.mask[indices] = True
        self.skeletons.mask[indices] = True
        self.scores.mask[indices] = True

    def num_valid(self):
        return np.sum(~self.scores.mask)


class _FinalResults(BaseResults):
    @classmethod
    def from_resolved(cls, resolved_results: _ResolvedResults):
        return _FinalResults(
            theta=resolved_results.theta.filled(np.nan),
            skeletons=resolved_results.skeletons.filled(np.nan),
            scores=resolved_results.scores.filled(np.nan),
        )

    @classmethod
    def from_shuffled(cls, shuffled_results: ShuffledResults):
        return _FinalResults(
            theta=np.full_like(shuffled_results.theta[:, 0], np.nan),
            skeletons=np.full_like(shuffled_results.scores[:, 0], np.nan),
            scores=np.full_like(shuffled_results.skeletons[:, 0], np.nan),
        )


def _make_continuous_partitions(
    shuffled_results: ShuffledResults, score_threshold: float, frame_rate: float
) -> _PartitionedResults:
    time_window = max(1, int(frame_rate * CONTINOUS_SEGMENT_TIME_WINDOW_SEC))
    min_segment_size = max(1, int(frame_rate * MIN_SEGMENT_SIZE_SEC))

    partitioned_results = _PartitionedResults(shuffled_results)

    # discard low score frames early (use the maximum value of both scores for now)
    good_score_frames = np.where(ma.greater_equal(ma.max(shuffled_results.scores, axis=1), score_threshold))[0]

    for frame_index in good_score_frames:

        prev_theta = partitioned_results.theta[frame_index - min(time_window, frame_index) : frame_index, 0]

        # if there is a big gap > time_window we start a new partition, with a random value (0)
        if np.all(np.any(prev_theta.mask, axis=1)):
            partitioned_results.set_partition(frame_index=frame_index, partition=0, new_partition=True)
        # otherwise we look in the time_window close past the closest non nan frame see if we can continue the
        # partition as long as the values stay continuous
        else:
            last_valid_index = np.where(~np.any(prev_theta.mask, axis=1))[0][-1]
            dists = [
                angle_distance(shuffled_results.theta[frame_index, k, :], prev_theta[last_valid_index],)
                for k in range(2)
            ]
            partition = int(np.argmin(dists))
            if dists[partition] < CONTINUOUS_ANGLES_DIST_THRESHOLD:
                partitioned_results.set_partition(frame_index=frame_index, partition=partition)

    # discard short segments
    for cur_partition_indices in partitioned_results.get_segments():
        if len(cur_partition_indices) < min_segment_size:
            partitioned_results.mask(cur_partition_indices)

    return partitioned_results


def _align_segments_with_labels(segments, partitioned_skeletons, labelled_skeletons, min_labelled=5):
    """
    Match the head/tail alignment with the results of the classical tracking in each of the segments,
     if there is enough labelled data in the segment
    """
    segments_alignment = ma.masked_all((len(segments),), dtype=np.uint8)
    for segment_index, segment in enumerate(segments):
        segment_skeletons = labelled_skeletons[segment]
        non_nan_labelled = np.any(~np.isnan(segment_skeletons), axis=(1, 2))
        labels_count = np.sum(non_nan_labelled)
        non_masked = ~np.any(partitioned_skeletons[segment].mask, axis=(1, 2, 3))
        to_compare = np.logical_and(non_nan_labelled, non_masked)

        similarity_scores = []
        for label_skel, partitioned_skeleton in zip(
            segment_skeletons[to_compare], partitioned_skeletons[segment][to_compare]
        ):
            dists = [skeleton_distance(label_skel, x) for x in partitioned_skeleton]
            similarity_scores.append(dists)

        if len(similarity_scores) > 0:
            mean_similarity_scores = np.mean(similarity_scores, axis=0)
            if mean_similarity_scores[0] * mean_similarity_scores[1] < 0 and labels_count > min_labelled:
                segments_alignment[segment_index] = np.argmax(mean_similarity_scores)

    return segments_alignment


def _calculate_smallest_gap_to_adjacent(segment_index, segments, segments_alignment):
    # evaluate how far away this segment is from known values
    score = np.nan
    segment_offset = np.nan
    if segment_index - 1 >= 0 and not segments_alignment.mask[segment_index - 1]:
        gap = segments[segment_index][0] - segments[segment_index - 1][-1]
        score = gap
        segment_offset = -1
    if segment_index + 1 < len(segments_alignment) and not segments_alignment.mask[segment_index + 1]:
        gap = segments[segment_index + 1][0] - segments[segment_index][-1]
        if np.isnan(score) or gap < score:
            score = gap
            segment_offset = 1

    return score, segment_offset


def _align_unlabelled_segments_with_adjacents(segments, segments_alignment, partitioned_skeletons, frame_rate: float):
    """
    Resolve the unaligned segments by comparing with adjacent segments,
    starting with the segments that have the least frames gap between an adjacent trusted segment
    Don't align isolated segments which a big gap between trusted segments
    """
    maximum_gap_allowed = max(1, int(frame_rate * MAXIMUM_GAP_ALLOWED_WITH_ADJACENT_SEGMENT_SEC))
    # ensure that if no segments have been aligned at all, pick one solution randomly to start
    if np.all(segments_alignment.mask):
        logger.info("There are no trusted segments with head decision to resolve the whole video, stopping analysis.")
        return segments_alignment

    # fix in priority the segments with known adjacent frames with little gap
    # until all segments are aligned except the isolated ones (further than maximum_gap_allowed)
    unaligned = np.where(segments_alignment.mask)[0]
    while len(unaligned) > 0:
        # we first pick the best candidate segment to align (there are known frames nearby before or after or both)
        all_gaps = [
            _calculate_smallest_gap_to_adjacent(
                segment_index=x, segments=segments, segments_alignment=segments_alignment,
            )
            for x in unaligned
        ]
        segment_to_fix_index = np.nanargmin(all_gaps, axis=0)[0]
        gap_to_adjacent_segment, adjacent_segment_offset = all_gaps[segment_to_fix_index]

        # abort if only isolated segments are left
        if gap_to_adjacent_segment > maximum_gap_allowed:
            break

        cur_segment_index = unaligned[segment_to_fix_index]
        cur_segment_skeleton = partitioned_skeletons[segments[cur_segment_index]]

        adjacent_segment_index = cur_segment_index + adjacent_segment_offset
        adjacent_alignment = segments_alignment[adjacent_segment_index]
        adjacent_segment = segments[adjacent_segment_index]
        adjacent_segment_skeleton = partitioned_skeletons[adjacent_segment][:, adjacent_alignment]

        if adjacent_segment_offset == -1:
            closest_unaligned_skeleton = cur_segment_skeleton[0]  # first frame of cur segment
            closest_known_skeleton = adjacent_segment_skeleton[-1]  # last frame of prev segment
        elif adjacent_segment_offset == 1:
            closest_unaligned_skeleton = cur_segment_skeleton[-1]  # last frame of cur segment
            closest_known_skeleton = adjacent_segment_skeleton[0]  # first frame of next segment
        else:
            raise ValueError()

        dists = [skeleton_distance(closest_known_skeleton, skel) for skel in closest_unaligned_skeleton]
        segments_alignment[cur_segment_index] = int(np.argmax(dists))

        unaligned = np.where(segments_alignment.mask)[0]

    return segments_alignment


def _init_unified_series(mixed_series):
    return ma.masked_all((mixed_series.shape[0],) + mixed_series.shape[2:], dtype=mixed_series.dtype)


def resolve_head_tail(
    shuffled_results: ShuffledResults, original_results: OriginalResults, frame_rate: float, score_threshold,
) -> BaseResults:
    len_series = len(shuffled_results)

    # Create continuous segments without jumps
    partitioned_results = _make_continuous_partitions(
        score_threshold=score_threshold, frame_rate=frame_rate, shuffled_results=shuffled_results,
    )
    segments = partitioned_results.get_segments()

    if len(segments) == 0:
        logger.error(
            f"Couldn't find any continuous segments of predicted data above the threshold {score_threshold},"
            f" stopping analysis."
        )
        return _FinalResults.from_shuffled(shuffled_results)

    # Choose each segment global alignment by comparing with labelled data
    segments_alignment = _align_segments_with_labels(
        segments, partitioned_results.skeletons, original_results.skeletons
    )

    # Fix unaligned segments here by comparing skeletons with neighboring segments iteratively
    segments_alignment = _align_unlabelled_segments_with_adjacents(
        segments, segments_alignment, partitioned_results.skeletons, frame_rate
    )

    # Compile results
    resolved_results = _ResolvedResults(partitioned_results)
    for segment, segment_alignment in zip(segments, segments_alignment):
        if not ma.is_masked(segment_alignment):
            resolved_results.resolve(segment, segment_alignment)

    # Filter the final results again by score threshold
    low_scores_indices = np.where(ma.masked_less(resolved_results.scores, score_threshold).mask)[0]
    resolved_results.mask(low_scores_indices)

    num_success = resolved_results.num_valid()
    original_num_success = np.any(~np.isnan(original_results.skeletons), axis=(1, 2)).sum()
    logger.info(
        f"Resolved head/tail, {num_success} out of {len_series} frames analyzed successfully "
        f"({float(num_success) / len_series * 100:.1f}%) (original features : {original_num_success}"
        f" or {(float(original_num_success) / len_series * 100):.1f}% of total)"
    )
    if num_success < original_num_success:
        logger.warning(f"Original results had {original_num_success - num_success} more successfully analyzed frames!")

    return _FinalResults.from_resolved(resolved_results)
