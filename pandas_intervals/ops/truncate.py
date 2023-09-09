import numpy as np
import pandas as pd

from .combine import combine
from .intersection import _get_mask_no_ref_overlap
from pandas_intervals.utils import apply_accessor


def ffill(arr: np.ndarray) -> np.ndarray:
    axis = 0
    _mask = np.isnan(arr)
    _idx = np.where(~_mask, np.arange(_mask.shape[axis]), 0)
    np.maximum.accumulate(_idx, axis=axis, out=_idx)
    return arr[_idx]


def bfill(arr):
    return ffill(arr[::-1])[::-1]


def ffill_step(arr: np.ndarray, increase: bool = True) -> np.ndarray:
    _mask = np.isnan(arr)
    _idx = np.where(~_mask, np.arange(len(_mask)), 0)
    np.maximum.accumulate(_idx, axis=0, out=_idx)
    return arr[_idx] + (2 * increase - 1) * (np.arange(len(arr)) - _idx)


def bfill_step(arr, increase: bool = False):
    return ffill_step(arr[::-1], increase)[::-1]


@apply_accessor
def truncate(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    if len(df_a) == 0 or len(df_b) == 0:
        return df_a

    df_b = df_b[["start", "end"]]
    df_b = combine(df_b)
    _mask = ((df_b["end"] - df_b["start"]) == 0) & (
        df_b["start"].isin(df_a["start"]) | df_b["start"].isin(df_a["end"])
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
    depth_ends_a = np.searchsorted(starts_b, ends_a) - np.searchsorted(
        starts_b, starts_a
    )
    depth_starts_a = np.searchsorted(ends_b, ends_a) - np.searchsorted(ends_b, starts_a)
    # Find nearest index
    idxs_starts_b_before_ends_a = np.searchsorted(starts_b, ends_a) - 1
    idxs_ends_b_after_starts_a = np.searchsorted(ends_b, starts_a)

    if max(depth_ends_a) == 0:
        ends_a_to_starts_b = np.zeros((0, 2))
    else:
        _shape = (
            np.sum(depth_ends_a).astype(int),
            2,
        )
        ends_a_to_starts_b = np.full(_shape, np.nan)  # cols: idxs_ends_a, idxs_starts_b
        _idxs = np.cumsum(depth_ends_a[depth_ends_a > 0]) - 1
        ends_a_to_starts_b[_idxs, 0] = np.arange(len(depth_ends_a))[
            depth_ends_a.nonzero()[0]
        ]
        ends_a_to_starts_b[:, 0] = bfill(ends_a_to_starts_b[:, 0])
        ends_a_to_starts_b[_idxs, 1] = idxs_starts_b_before_ends_a[
            depth_ends_a.nonzero()[0]
        ]
        ends_a_to_starts_b[:, 1] = bfill_step(ends_a_to_starts_b[:, 1])
        # Link col1 with starts_b (location)
        ends_a_to_starts_b[:, 1] = starts_b[ends_a_to_starts_b[:, 1].astype(int)]

    if max(depth_starts_a) == 0:
        starts_a_to_ends_b = np.zeros((0, 2))
    else:
        _shape = (
            np.sum(depth_starts_a).astype(int),
            2,
        )
        starts_a_to_ends_b = np.full(_shape, np.nan)  # cols: idxs_starts_a, idxs_ends_b
        _idxs = np.cumsum(depth_starts_a[depth_starts_a > 0]) - 1
        starts_a_to_ends_b[_idxs, 0] = np.arange(len(depth_starts_a))[
            depth_starts_a.nonzero()[0]
        ]
        starts_a_to_ends_b[:, 0] = bfill(starts_a_to_ends_b[:, 0])
        starts_a_to_ends_b[_idxs, 1] = idxs_ends_b_after_starts_a[
            depth_starts_a.nonzero()[0]
        ]
        starts_a_to_ends_b[:, 1] = bfill_step(
            starts_a_to_ends_b[:, 1], increase=True
        )  # TODO verify increase or decrease
        # Link col1 with ends_b (location)
        starts_a_to_ends_b[:, 1] = ends_b[starts_a_to_ends_b[:, 1].astype(int)]

    ## Drop starts_a/ends_a that are overlapped by B
    # if these indices are equal, then these points don't overlap into a B interval
    mask_starts_a_non_overlap_b = np.searchsorted(
        starts_b, starts_a
    ) - 1 != np.searchsorted(ends_b, starts_a)
    mask_ends_a_non_overlap_b = np.searchsorted(
        starts_b, ends_a
    ) - 1 != np.searchsorted(ends_b, ends_a)

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
    return pd.concat([df_result, df_a_no_overlap])
