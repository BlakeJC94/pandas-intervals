from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .combine import intervals_combine
from .intersection import _get_mask_no_ref_overlap
from pandas_intervals.vis import plot_interval_groups as plt


def _points_from_intervals(
    interval_groups: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
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
def _intervals_difference(
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
    axis = 0
    _mask = np.isnan(arr)
    _idx = np.where(~_mask, np.arange(_mask.shape[axis]), 0)
    np.maximum.accumulate(_idx, axis=axis, out=_idx)
    return arr[_idx]


def bfill(arr):
    return ffill(arr[::-1])[::-1]


def ffill_step(arr: np.ndarray, increase=True) -> np.ndarray:
    _mask = np.isnan(arr)
    _idx = np.where(~_mask, np.arange(len(_mask)), 0)
    np.maximum.accumulate(_idx, axis=0, out=_idx)
    return arr[_idx] + (2 * increase - 1) * (np.arange(len(arr)) - _idx)


def bfill_step(arr, increase=False):
    return ffill_step(arr[::-1], increase)[::-1]


# TODO Drop points in B that match start/ends in A (no effect)
def intervals_difference(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    if len(df_a) == 0 or len(df_b) == 0:
        return df_a

    df_b = df_b[["start", "end"]]
    df_b = intervals_combine(df_b)
    _mask = (
        ((df_b['end'] - df_b['start']) == 0)
        & (
            df_b['start'].isin(df_a['start'])
            | df_b['start'].isin(df_a['end'])
        )
    )
    df_b = df_b[~_mask]
    df_b = df_b.sort_values("start")

    if len(df_b) == 0:
        return df_a

    # Filter out intervals in A that don't overlap with B
    _mask = _get_mask_no_ref_overlap(df_b, df_a)
    df_a_no_overlap = df_a[_mask]
    df_a = df_a[~_mask]

    if len(df_a) == 0:
        return df_a_no_overlap

    starts_b, ends_b = df_b["start"].to_numpy(), df_b["end"].to_numpy()  # SORTED
    starts_a, ends_a = df_a["start"].to_numpy(), df_a["end"].to_numpy()

    # Get depth
    depth_ends_a = np.searchsorted(starts_b, ends_a) - np.searchsorted(starts_b, starts_a)
    depth_starts_a = np.searchsorted(ends_b, ends_a) - np.searchsorted(ends_b, starts_a)
    # Find nearest index
    idxs_starts_b_before_ends_a = np.searchsorted(starts_b, ends_a) - 1
    idxs_ends_b_after_starts_a = np.searchsorted(ends_b, starts_a)

    if max(depth_ends_a) == 0:
        ends_a_to_starts_b = np.zeros((0, 2))
    else:
        _shape = ( np.sum(depth_ends_a).astype(int), 2,)
        ends_a_to_starts_b = np.full(_shape, np.nan)  # cols: idxs_ends_a, idxs_starts_b
        _idxs = np.cumsum(depth_ends_a[depth_ends_a>0]) - 1
        ends_a_to_starts_b[_idxs, 0] = np.arange(len(depth_ends_a))[depth_ends_a.nonzero()[0]]
        ends_a_to_starts_b[:, 0] = bfill(ends_a_to_starts_b[:, 0])
        ends_a_to_starts_b[_idxs, 1] = idxs_starts_b_before_ends_a[depth_ends_a.nonzero()[0]]
        ends_a_to_starts_b[:, 1] = bfill_step(ends_a_to_starts_b[:, 1])
        # Link col1 with starts_b (location)
        ends_a_to_starts_b[:, 1] = starts_b[ends_a_to_starts_b[:, 1].astype(int)]

    if max(depth_starts_a) == 0:
        starts_a_to_ends_b = np.zeros((0, 2))
    else:
        _shape = ( np.sum(depth_starts_a).astype(int), 2,)
        starts_a_to_ends_b = np.full(_shape, np.nan)  # cols: idxs_starts_a, idxs_ends_b
        _idxs = np.cumsum(depth_starts_a[depth_starts_a>0]) - 1
        starts_a_to_ends_b[_idxs, 0] = np.arange(len(depth_starts_a))[depth_starts_a.nonzero()[0]]
        starts_a_to_ends_b[:, 0] = bfill(starts_a_to_ends_b[:, 0])
        starts_a_to_ends_b[_idxs, 1] = idxs_ends_b_after_starts_a[depth_starts_a.nonzero()[0]]
        starts_a_to_ends_b[:, 1] = bfill_step(starts_a_to_ends_b[:, 1], increase=True)  # TODO verify increase or decrease
        # Link col1 with ends_b (location)
        starts_a_to_ends_b[:, 1] = ends_b[starts_a_to_ends_b[:, 1].astype(int)]


    ## Drop starts_a/ends_a that are overlapped by B
    # if these indices are equal, then these points don't overlap into a B interval
    mask_starts_a_non_overlap_b = np.searchsorted(starts_b, starts_a) - 1 != np.searchsorted(ends_b, starts_a)
    mask_ends_a_non_overlap_b = np.searchsorted(starts_b, ends_a) - 1 != np.searchsorted(ends_b, ends_a)

    ## Construct resulting intervals from points
    n_rows_a = len(df_a)
    starts_a = np.stack(
        [
            np.arange(n_rows_a),
            starts_a,
        ],
        axis=-1,
    )
    ends_a = np.stack(
        [
            np.arange(n_rows_a),
            ends_a,
        ],
        axis=-1,
    )
    points_result = np.concatenate(
        [
            starts_a[mask_starts_a_non_overlap_b],
            ends_a[mask_ends_a_non_overlap_b],
            ends_a_to_starts_b,
            starts_a_to_ends_b,
        ],
        axis=0,
    )
    idxs_results_sort = np.lexsort((points_result[:, 1], points_result[:, 0]))
    points_result = points_result[idxs_results_sort]
    idxs_a, starts_result, ends_result = (
        points_result[::2, 0],
        points_result[::2, 1],
        points_result[1::2, 1],
    )
    idxs_a = idxs_a.astype(int)

    df_result = pd.DataFrame(
        dict(start=starts_result, end=ends_result), index=df_a.index[idxs_a]
    )
    df_result = pd.concat([df_result, df_a.iloc[idxs_a, 2:]], axis=1)
    # Re-join points in A from start of func
    df_result = pd.concat([df_result, df_a_no_overlap])

    return df_result
