from itertools import permutations
from typing import Callable, Iterable, Union, List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd

from .interval_utils import (
    _get_overlapping_mask,
    _atomize_intervals,
)
from tests.helpers import (
    combine_basic,
    intersection_basic,
    complement_basic,
)


def intervals_union(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs, axis=0).drop_duplicates()


# TODO Upgrade to vectorised version
# np.searchsorted
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
        mask_no_overlap_ref = (start_insert_idxs == end_insert_idxs)
        results.append(df[mask_no_overlap_ref])

    breakpoint()
    return pd.concat(results, axis=0)
    # return intersection_basic(df_a, df_b)


# TODO Upgrade to vectorised version
def intervals_complement(
    df: pd.DataFrame,
    aggregations: Optional[Dict[str, Union[str, Callable]]] = None,
    left_bound: Optional[float] = None,
    right_bound: Optional[float] = None,
):
    # df = intervals_combine(df, aggregations)
    return complement_basic(
        df,
        aggregations,
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


# TODO Upgrade to vectorised version
def intervals_combine(
    df: pd.DataFrame,
    aggregations: Optional[Dict[str, Union[str, Callable]]] = None,
):
    return combine_basic(df, aggregations)


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
