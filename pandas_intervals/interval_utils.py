from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEBUG = False


def _get_overlapping_mask(df: pd.DataFrame) -> np.ndarray:
    # Split out starts and ends, append interval index column
    # Append +1 col to the starts and -1 col to the ends
    n_rows = len(df)
    idxs_intervals = np.arange(n_rows)
    starts = np.stack(
        [idxs_intervals, df["start"].to_numpy(), np.ones(n_rows)], axis=-1
    )
    ends = np.stack([idxs_intervals, df["end"].to_numpy(), -np.ones(n_rows)], axis=-1)

    # Concat and sort by time
    cap = np.array([[-1, -np.inf, 0]])
    times = np.concatenate([cap, starts, ends], axis=0)
    idxs_times_sort = np.lexsort((times[:, 2], times[:, 1]))
    times = times[idxs_times_sort]

    # Csum step column
    result = np.cumsum(times[:, 2])

    # Get idxs where result is 1
    mask_result_1 = result == 1
    (idxs_result_1,) = mask_result_1.nonzero()  # Which idxs are the start of new group?
    idxs_overlap_groups = idxs_result_1

    # Check indices before result==1 and filter to those which are equal to 0
    idxs_result_1_prev = idxs_result_1 - 1
    check_idxs_results_1_prev_down = result[idxs_result_1_prev] < 1
    idxs_result_1_prev_down = idxs_result_1_prev[check_idxs_results_1_prev_down]
    idxs_overlap_groups = np.intersect1d(
        idxs_overlap_groups, idxs_result_1_prev_down + 1
    )

    # Check indices before result==1 and filter to those which are equal to 0
    idxs_result_1_next = idxs_result_1 + 1
    check_idxs_results_1_next_up = result[idxs_result_1_next] > 1
    idxs_result_1_next_up = idxs_result_1_next[check_idxs_results_1_next_up]
    idxs_overlap_groups = np.intersect1d(idxs_overlap_groups, idxs_result_1_next_up - 1)

    # Get integer mask of results of different overlap groups
    mask_overlap_groups = np.zeros(len(result), dtype=int)
    mask_overlap_groups[idxs_overlap_groups] = 1
    mask_overlap = np.cumsum(mask_overlap_groups)
    mask_overlap *= mask_overlap_groups | (result > 1)

    # Get indices and groups from result mask
    idxs_overlap = np.stack(
        [
            times[(mask_overlap > 0), 0],
            mask_overlap[mask_overlap > 0],
        ],
        axis=-1,
    )
    mask_result = np.zeros(len(df))
    mask_result[idxs_overlap[:, 0].astype(int)] = idxs_overlap[:, 1]
    return mask_result


def _points_from_intervals(interval_groups: List[np.ndarray]) -> Tuple[np.ndarray]:
    n_interval_groups = len(interval_groups)
    interval_points, interval_indices = [], []
    for i, intervals in enumerate(interval_groups):
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

    return atomized_intervals, interval_idxs
