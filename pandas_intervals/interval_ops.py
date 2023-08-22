from itertools import permutations
from typing import Callable, Union, List, Dict, Optional

import numpy as np
import pandas as pd

from .interval_utils import (
    _get_overlapping_mask,
    _atomize_intervals,
)

def intervals_union(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs, axis=0).drop_duplicates()


def intervals_intersection(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> pd.DataFrame:
    results = []
    for df_ref, df in permutations([df_a, df_b]):
        ref_starts = np.sort(df_ref["start"].values)
        ref_ends = np.sort(df_ref["end"].values)
        starts = df["start"].values
        ends = df["end"].values

        # Find the index at which label starts/ends would be inserted in the
        # ref_labels starts/ends
        start_insert_idxs = np.searchsorted(ref_ends, starts)
        end_insert_idxs = np.searchsorted(ref_starts, ends)

        # When the insertion index is the same for both the interval start and end, the
        # interval has no overlapping intervals in the reference set
        mask_no_overlap_ref = start_insert_idxs == end_insert_idxs
        results.append(df[~mask_no_overlap_ref])

    return pd.concat(results, axis=0)


def intervals_complement(
    df: pd.DataFrame,
    aggregations: Optional[Dict[str, Union[str, Callable]]] = None,
    left_bound: Optional[float] = None,
    right_bound: Optional[float] = None,
):
    if len(df) == 0:
        return df

    df = intervals_combine(df, aggregations=aggregations).sort_values("start")

    (start_first, end_first), metadata_first = df.iloc[0, :2], df.iloc[0, 2:]
    (_start_last, end_last), metadata_last = df.iloc[-1, :2], df.iloc[-1, 2:]
    if left_bound is None:
        left_bound = end_first
    if right_bound is None:
        right_bound = end_last

    edges: List[pd.Series] = []
    if left_bound < start_first:
        left_edge = metadata_first
        left_edge["start"] = left_bound
        left_edge["end"] = start_first
        left_edge = left_edge[df.columns]
        edges.append(left_edge)
    if end_last < right_bound:
        right_edge = metadata_last
        right_edge["start"] = end_last
        right_edge["end"] = right_bound
        right_edge = right_edge[df.columns]
        edges.append(right_edge)

    df_c = df.rolling(2).agg(aggregations).iloc[1:]
    df_c["start"] = df.iloc[:-1, 1].to_numpy()
    df_c["end"] = df.iloc[1:, 0].to_numpy()
    df_c = df_c[df.columns]

    return pd.concat([pd.DataFrame(edges), df_c], axis=0).sort_values("start")


def intervals_overlap(df: pd.DataFrame):
    if df.empty:
        return df
    return df.loc[(_get_overlapping_mask(df) > 0)]


def intervals_non_overlap(df: pd.DataFrame):
    if df.empty:
        return df
    return df.loc[~(_get_overlapping_mask(df) > 0)]


def intervals_combine(
    df: pd.DataFrame,
    aggregations: Optional[Dict[str, Union[str, Callable]]] = None,
):
    if df.empty:
        return df

    aggregations = aggregations or {"start": "min", "end": "max"}
    aggregations.update({c: "first" for c in df.columns if c not in aggregations})
    df_sorted = df.sort_values("start")

    # Filter out non_overlapping intervals to those that overlap
    overlap_mask = _get_overlapping_mask(df_sorted)
    df_sorted_non_overlap = df_sorted.loc[overlap_mask == 0]
    df_sorted_overlap = df_sorted.loc[overlap_mask > 0]

    # Aggregate overlaps
    group_inds = overlap_mask[overlap_mask > 0]
    df_sorted_overlap_agg = df_sorted_overlap.groupby(group_inds).agg(aggregations)

    return pd.concat([df_sorted_non_overlap, df_sorted_overlap_agg], axis=0)


def intervals_difference(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    aggregations: Optional[List[str]] = None,
):
    if len(df_a) == 0 or len(df_b) == 0:
        return df_a

    df_b = intervals_combine(df_b, aggregations)
    df_a = intervals_combine(df_a, aggregations)

    input_columns = df_a.columns
    intervals_a_metadata = df_a.drop(["start", "end"], axis=1)

    intervals_a = df_a[["start", "end"]].to_numpy()
    intervals_b = df_b[["start", "end"]].to_numpy()

    atoms, indices = _atomize_intervals(
        [intervals_a, intervals_b],
        drop_gaps=False,
    )
    mask_a_atoms = (indices[:, 0] != -1) & (indices[:, 1] == -1)
    result, indices = atoms[mask_a_atoms], indices[mask_a_atoms, 0]

    intervals_a_diff_b = intervals_a_metadata.iloc[indices].reset_index(drop=True)
    intervals_a_diff_b[["start", "end"]] = result
    return intervals_a_diff_b[input_columns]
