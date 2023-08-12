from typing import Callable, Iterable, Union, List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd

from .interval_utils import (
    _get_overlapping_mask,
    _atomize_intervals,
)
from tests.helpers import (
    intersection_basic,
    complement_basic,
)


def intervals_union(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs, axis=0).drop_duplicates()


# TODO Upgrade to vectorised version
def intervals_intersection(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    df_a, df_b = dfs[0], intervals_union(*dfs[1:])
    return intersection_basic(df_a, df_b)


# TODO Upgrade to vectorised version
def intervals_complement(
    dfs: List[pd.DataFrame],
    left_bound: Optional[float] = None,
    right_bound: Optional[float] = None,
):
    df = intervals_combine(dfs)
    return complement_basic(
        df,
        left_bound,
        right_bound,
    )


# TODO Upgrade to vectorised version
def intervals_overlap(
    df: pd.DataFrame,
):
    if df.empty:
        return df
    return df.loc[_get_overlapping_mask(df)]


# TODO Upgrade to vectorised version
def intervals_non_overlap(
    df: pd.DataFrame,
):
    if df.empty:
        return df
    return df.loc[~_get_overlapping_mask(df)]


def intervals_combine(
    dfs: List[pd.DataFrame],
    groupby_cols: Optional[List[str]] = None,
    aggregations: Optional[Dict[str, Union[str, Callable]]] = None,
):
    intervals = intervals_union(dfs, groupby_cols)
    if intervals.empty:
        return intervals

    if aggregations is None:
        aggregations = {}

    aggregations = {c: "first" for c in intervals.columns if c not in aggregations}

    combined_labels = []
    for _, interval_group in intervals.groupby(groupby_cols, as_index=False):
        interval_group_sorted = interval_group.sort_values("start")

        # TODO Vectorise this
        # Loop over labels and compare each to the previous label to find labels to combine
        group_inds = []
        ind, interval_end_time = 0, 0
        for start, end in interval_group_sorted[["start", "end"]].values:
            # If interval is within previous label, combine them
            if start <= interval_end_time:
                interval_end_time = max(interval_end_time, end)
                group_inds.append(ind)
            # If not, start a new interval
            else:
                interval_end_time = end
                ind += 1
                group_inds.append(ind)

        grpd_labels = interval_group_sorted.groupby(group_inds).agg(aggregations)
        combined_labels.append(grpd_labels)

    return pd.concat(combined_labels).reset_index(drop=True)


def intervals_difference(
    df: pd.DataFrame,
    dfs: List[pd.DataFrame],
    groupby_cols: Optional[List[str]] = None,
    min_len: Optional[float] = None,
):
    if len(dfs) == 0:
        return df

    intervals_b = intervals_combine(dfs, groupby_cols)
    if len(intervals_b) == 0:
        return df

    intervals_a = intervals_combine([df], groupby_cols)

    # TODO Fix groupby to select vals in intervals_b
    results = []
    for _, intervals_a_group in intervals_a.groupby(groupby_cols):
        input_columns = df.columns
        intervals_a_metadata = df.drop(["start", "end"], axis=1)

        intervals_a = intervals_a[["start", "end"]].values.copy()
        intervals_b = intervals_b[["start", "end"]].values.copy()

        atoms, indices = _atomize_intervals(
            [intervals_a, intervals_b],
            drop_gaps=False,
            min_len=min_len,
        )
        mask_a_atoms = (indices[:, 0] != -1) & (indices[:, 1] == -1)
        result, indices = atoms[mask_a_atoms], indices[mask_a_atoms, 0]

        intervals_a_diff_b = intervals_a_metadata.iloc[indices].reset_index(drop=True)
        intervals_a_diff_b[["start", "end"]] = result
        results.append(intervals_a_diff_b[input_columns])

    return intervals_union(result, groupby_cols)
