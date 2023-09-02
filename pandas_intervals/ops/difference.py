from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .combine import intervals_combine
from .intersection import _get_mask_no_ref_overlap


def _points_from_intervals(interval_groups: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
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


# TODO Find a way to remove assumption for flat A
def intervals_difference(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    aggregations: Optional[List[str]] = None,
):
    if len(df_a) == 0 or len(df_b) == 0:
        return df_a

    df_b = intervals_combine(df_b)
    df_a = intervals_combine(df_a, aggregations)

    input_columns = df_a.columns
    intervals_a_metadata = df_a.drop(["start", "end"], axis=1)

    intervals_a = df_a[["start", "end"]].to_numpy()
    intervals_b = df_b[["start", "end"]].to_numpy()

    # Filter out 0-dur intervals in B that equal start/end of an interval in A
    mask_intervals_b_0_dur = (intervals_b[:, 0] == intervals_b[:, 1]) & (
        np.isin(intervals_b[:, 0], intervals_a[:, 0])
        | np.isin(intervals_b[:, 0], intervals_a[:, 1])
    )
    intervals_b = intervals_b[~mask_intervals_b_0_dur]

    atoms, indices = _atomize_intervals(
        [intervals_a, intervals_b],
        drop_gaps=False,
    )
    mask_a_atoms = (indices[:, 0] != -1) & (indices[:, 1] == -1)
    result, indices = atoms[mask_a_atoms], indices[mask_a_atoms, 0]

    intervals_a_diff_b = intervals_a_metadata.iloc[indices].reset_index(drop=True)
    intervals_a_diff_b[["start", "end"]] = result
    return intervals_a_diff_b[input_columns]


def ffill(arr: np.ndarray) -> np.ndarray:
    axis=0
    _mask = np.isnan(arr)
    _idx = np.where(~_mask, np.arange(_mask.shape[axis]), 0)
    np.maximum.accumulate(_idx, axis=axis, out=_idx)
    return arr[_idx]

def bfill(arr):
    return ffill(arr[::-1])[::-1]

def ffill_step(arr: np.ndarray, increase = True) -> np.ndarray:
    axis=0
    _mask = np.isnan(arr)
    _idx = np.where(~_mask, np.arange(_mask.shape[axis]), 0)
    np.maximum.accumulate(_idx, axis=axis, out=_idx)
    floob = np.concatenate([[0], (np.cumsum(_idx[1:] == _idx[:-1])) * (_idx[1:] == _idx[:-1])])
    return arr[_idx] + (2 * increase - 1) * floob

def bfill_step(arr, increase=False):
    return ffill_step(arr[::-1], increase)[::-1]

def intervals_difference_v2(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    df_b = df_b[["start", "end"]]
    df_b = intervals_combine(df_b)

    # TODO Filter out points in A and handle separately

    # Filter out intervals in A that don't overlap with B
    _mask = _get_mask_no_ref_overlap(df_b, df_a)
    df_a_no_overlap = df_a[_mask]  # TODO Join with results at end
    df_a = df_a[~_mask]

    ## Get the indices and depth of points of A
    # Split out starts and ends, append interval index column
    # Append +1 col to the starts and -1 col to the ends
    n_rows_a = len(df_a)
    idxs_intervals = np.arange(n_rows_a)
    starts_a = np.stack(
        [
            idxs_intervals,
            df_a["start"].to_numpy(),
            np.ones(n_rows_a),
        ],
        axis=-1,
    )
    starts_a = starts_a[np.argsort(starts_a[:, 1])]
    ends_a = np.stack(
        [
            idxs_intervals,
            df_a["end"].to_numpy(),
            -np.ones(n_rows_a),
        ],
        axis=-1,
    )
    ends_a = ends_a[np.argsort(ends_a[:, 1])]

    # Sort points of A
    points_a = np.concatenate([starts_a, ends_a], axis=0)
    idxs_times_sort = np.lexsort((points_a[:, 2], points_a[:, 1]))
    points_a = points_a[idxs_times_sort]

    # Calculate depth sequences of A
    depth_a_left = np.cumsum(points_a[:, 2])
    depth_a_right = -np.cumsum(points_a[::-1, 2])[::-1]

    ## Find the starts_a that need to be copied to ends_b
    # Get the indices of starts_a where ends_b come after
    ends_b = df_b["end"].to_numpy()
    idxs_starts_a_left_of_ends_b = np.searchsorted(starts_a[:, 1], ends_b) - 1  # ~ ends_b
    idxs_points_a_left_of_ends_b = np.searchsorted(points_a[:, 1], ends_b) - 1  # ~ ends_b
    # Filter out ends_b that occur left of A
    _mask = (idxs_starts_a_left_of_ends_b > -1) & (idxs_points_a_left_of_ends_b > -1)
    idxs_starts_a_left_of_ends_b = idxs_starts_a_left_of_ends_b[_mask]
    idxs_points_a_left_of_ends_b = idxs_points_a_left_of_ends_b[_mask]
    ends_b = ends_b[_mask]
    # Get the depth at these points
    depth_ends_b = depth_a_left[idxs_points_a_left_of_ends_b]  # ~ ends_b
    # Drop ends_b with zero depth
    _mask = (depth_ends_b > 0)
    ends_b = ends_b[_mask]
    depth_ends_b = depth_ends_b[_mask]

    ## Copy starts_a to ends_b
    if len(ends_b) == 0:
        full_starts_a_left_of_ends_b = np.zeros((0, 3))
    else:
        # Initialise sequence of start_a points that are copied to ends_b
        _shape = (np.sum(depth_ends_b).astype(int), 2)  # ~ (starts_a indices, ends_b times)
        full_starts_a_left_of_ends_b = np.full(_shape, np.nan)
        # Backfill ends_b times column
        _idxs = np.cumsum(depth_ends_b).astype(int) - 1
        full_starts_a_left_of_ends_b[_idxs, 0] = idxs_starts_a_left_of_ends_b
        full_starts_a_left_of_ends_b[_idxs, 1] = ends_b
        full_starts_a_left_of_ends_b[:, 1] = bfill(full_starts_a_left_of_ends_b[:, 1])
        # Backfill and decrease previous starts_a indices
        full_starts_a_left_of_ends_b[:, 0] = bfill_step(
            full_starts_a_left_of_ends_b[:, 0],
            increase=False,
        )
        # Assign indices
        _idxs = full_starts_a_left_of_ends_b[:, 0].astype(int)
        full_starts_a_left_of_ends_b[:, 0] = starts_a[_idxs, 0]
        full_starts_a_left_of_ends_b = np.concatenate(
            [
                full_starts_a_left_of_ends_b,
                np.ones((len(full_starts_a_left_of_ends_b), 1))
            ],
            axis=1,
        )


    ## Find the ends_a that need to be copied to starts_b
    # Get the indices of ends_a where starts_b come after
    starts_b = df_b["start"].to_numpy()
    idxs_ends_a_right_of_starts_b = np.searchsorted(ends_a[:, 1], starts_b)  # ~ starts_b
    idxs_points_a_right_of_starts_b = np.searchsorted(points_a[:, 1], starts_b)  # ~ starts_b
    # Filter out starts_b that occur right of A
    _mask = (idxs_ends_a_right_of_starts_b < len(ends_a) - 1) & (idxs_points_a_right_of_starts_b < len(points_a)- 1)
    idxs_ends_a_right_of_starts_b = idxs_ends_a_right_of_starts_b[_mask]
    idxs_points_a_right_of_starts_b = idxs_points_a_right_of_starts_b[_mask]
    starts_b = starts_b[_mask]
    # Get the depth at these points
    depth_starts_b = depth_a_right[idxs_points_a_right_of_starts_b]  # ~ starts_b
    # Drop starts_b with zero depth
    _mask = (depth_starts_b > 0)
    starts_b = starts_b[_mask]
    depth_starts_b = depth_starts_b[_mask]

    ## Copy ends_a to starts_b
    if len(ends_b) == 0:
        full_ends_a_right_of_starts_b = np.zeros((0, 3))
    else:
        # Initialise sequence of start_a points that are copied to ends_b
        _shape = (np.sum(depth_starts_b).astype(int), 2)  # ~ (starts_a indices, ends_b times)
        full_ends_a_right_of_starts_b = np.full(_shape, np.nan)
        # Backfill ends_b times column
        _idxs = np.cumsum(depth_starts_b).astype(int) - 1
        full_ends_a_right_of_starts_b[_idxs, 0] = idxs_ends_a_right_of_starts_b
        full_ends_a_right_of_starts_b[_idxs, 1] = starts_b  # TODO Verify bfill ops here
        full_ends_a_right_of_starts_b[:, 1] = bfill(full_ends_a_right_of_starts_b[:, 1])
        # Backfill and decrease previous starts_a indices
        full_ends_a_right_of_starts_b[:, 0] = bfill_step(
            full_ends_a_right_of_starts_b[:, 0],
            increase=True,
        )
        # Assign indices
        _idxs = full_ends_a_right_of_starts_b[:, 0].astype(int)
        full_ends_a_right_of_starts_b[:, 0] = ends_a[_idxs, 0]
        full_ends_a_right_of_starts_b = np.concatenate(
            [
                full_ends_a_right_of_starts_b,
                -1 * np.ones((len(full_ends_a_right_of_starts_b), 1))
            ],
            axis=1,
        )

    ## Drop starts_a/ends_a that are overlapped by B
    starts_b = np.sort(df_b["start"].to_numpy())
    ends_b = np.sort(df_b["end"].to_numpy())
    # if these indices are equal, then these points don't overlap into a B interval
    mask_starts_a_non_overlap_b = np.searchsorted(starts_b, starts_a[:, 1]) - 1 != np.searchsorted(
        ends_b, starts_a[:, 1]
    )
    mask_ends_a_non_overlap_b = np.searchsorted(starts_b, ends_a[:, 1]) - 1 != np.searchsorted(
        ends_b, ends_a[:, 1]
    )

    ## Construct resulting intervals from points
    points_result = np.concatenate(
        [
            starts_a[mask_starts_a_non_overlap_b],
            ends_a[mask_ends_a_non_overlap_b],
            full_ends_a_right_of_starts_b,
            full_starts_a_left_of_ends_b,
        ],
        axis=0,
    )
    idxs_results_sort = np.lexsort((points_result[:, 1], points_result[:, 0]))
    points_result = points_result[idxs_results_sort]
    idxs_a, starts_result, ends_result = points_result[::2, 0], points_result[::2, 1], points_result[1::2,1]
    idxs_a = idxs_a.astype(int)

    df_result = pd.DataFrame(dict(start=starts_result, end=ends_result), index=df_a.index[idxs_a])
    df_result = pd.concat([df_result, df_a.iloc[idxs_a, 2:]], axis=1)
    # Re-join points in A from start of func
    df_result = pd.concat([df_result, df_a_no_overlap])

    return df_result
