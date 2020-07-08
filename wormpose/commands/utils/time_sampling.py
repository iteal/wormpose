from typing import Optional

import numpy as np

from wormpose.pose.results_datatypes import BaseResults


def _resample(values: Optional[np.ndarray], timestamp: np.ndarray) -> Optional[np.ndarray]:
    if values is None:
        return values
    values = np.array(values)
    resampled = np.full((timestamp[-1] - timestamp[0] + 1,) + values[0].shape, np.nan, values.dtype)
    for i in range(len(values)):
        t = timestamp[i] - timestamp[0]
        resampled[t] = values[i]
    return resampled


def resample_results(results: BaseResults, timestamp: np.ndarray):
    results.scores = _resample(results.scores, timestamp)
    results.theta = _resample(results.theta, timestamp)
    results.skeletons = _resample(results.skeletons, timestamp)
