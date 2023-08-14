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
def intervals_intersection(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    return intersection_basic(df_a, df_b)


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


def intervals_combine(
    df: List[pd.DataFrame],
    aggregations: Optional[Dict[str, Union[str, Callable]]] = None,
):
    return combine_basic(df, aggregations)


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
    return intervals_a_diff_b[input_columns]
