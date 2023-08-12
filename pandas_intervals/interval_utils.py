from typing import Union, List, Tuple, Optional

import numpy as np
import pandas as pd

from tests.helpers import _overlap_mask_basic


# TODO searchsorted implementation
def _get_overlapping_mask(df: pd.DataFrame) -> np.ndarray:
    return _overlap_mask_basic(df)


# TODO Remove duplication
def _intervals_overlapping(intervals: Union[np.ndarray, pd.DataFrame]):
    if isinstance(intervals, pd.DataFrame):
        intervals = intervals[["start", "end"]].values
    intervals = intervals[np.argsort(intervals[:, 0]), :]
    starts, ends = intervals[:, 0], intervals[:, 1]
    overlaps = starts[1:] - ends[:-1]
    return (overlaps < 0).any()


def _points_from_intervals(interval_groups: List[np.ndarray]) -> Tuple[np.ndarray]:
    n_interval_groups = len(interval_groups)
    interval_points, interval_indices = [], []
    for i, intervals in enumerate(interval_groups):
        assert not _intervals_overlapping(
            intervals
        ), "Expected the intervals within a group to be non-overlapping"
        n_intervals = len(intervals)

        indices = np.zeros((n_intervals, n_interval_groups))
        indices[:, i] = np.arange(n_intervals) + 1
        indices = np.concatenate([indices, -indices], axis=0)

        points = np.concatenate([intervals[:, 0:1], intervals[:, 1:2]], axis=0)

        interval_points.append(points)
        interval_indices.append(indices)

    interval_points = np.concatenate(interval_points, axis=0)
    interval_indices = np.concatenate(interval_indices, axis=0)

    idx = np.argsort(interval_points[:, 0])
    interval_points = interval_points[idx, :]
    interval_indices = interval_indices[idx, :]

    interval_indices = np.abs(np.cumsum(interval_indices, axis=0)) - 1
    return interval_points, interval_indices


def _atomize_intervals(
    interval_groups: List[np.ndarray],
    min_len: Optional[float] = None,
    drop_gaps: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    points, indices = _points_from_intervals(interval_groups)
    for i in range(1, len(interval_groups)):
        indices[indices[:, i] != -1, :i] = -1

    starts, ends = points[:-1, 0:1], points[1:, 0:1]
    interval_idxs = indices[:-1].astype(int)
    atomized_intervals = np.concatenate([starts, ends], axis=1)

    if drop_gaps:
        mask_nongap_intervals = (interval_idxs != -1).any(axis=1)

        atomized_intervals = atomized_intervals[mask_nongap_intervals]
        interval_idxs = interval_idxs[mask_nongap_intervals]

    if min_len is not None:
        interval_lengths = atomized_intervals[:, 1] - atomized_intervals[:, 0]
        mask_above_min_len = interval_lengths > min_len

        atomized_intervals = atomized_intervals[mask_above_min_len]
        interval_idxs = interval_idxs[mask_above_min_len]

    return atomized_intervals, interval_idxs
