from typing import Tuple

import pandas as pd

from pandas_intervals.utils import apply_accessor, df_to_list


def _intervals_intersect(
    ivl_a: Tuple[float, float], ivl_b: Tuple[float, float]
) -> bool:
    start_a, end_a, *_ = ivl_a
    start_b, end_b, *_ = ivl_b
    return (
        (start_a < start_b < end_a)
        or (start_a < end_b < end_a)
        or (start_b < start_a < end_b)
        or (start_b < end_a < end_b)
        or (start_a == start_b)
        or (end_a == end_b)
    )


@apply_accessor
def intersection(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    # Track the index as a column
    df_a = df_a.assign(_index=df_a.index)
    df_b = df_b.assign(_index=df_b.index)

    intervals_a = df_to_list(df_a.sort_values("start"))
    intervals_b = df_to_list(df_b.sort_values("start"))

    result = set()
    for ivl_a in intervals_a:
        start_a, end_a, *_ = ivl_a
        for ivl_b in intervals_b:
            start_b, end_b, *_ = ivl_a
            if end_b < start_a:
                continue
            if end_a < start_b:
                break

            if _intervals_intersect(ivl_a, ivl_b):
                result.add(ivl_a)
                result.add(ivl_b)

    return (
        pd.DataFrame(result, columns=df_a.columns)
        .set_index("_index")
        .rename_axis(index=None)
    )


@apply_accessor
def symdiff(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    # Track the index as a column
    df_a = df_a.assign(_index=df_a.index)
    df_b = df_b.assign(_index=df_b.index)

    intervals_a = df_to_list(df_a)
    intervals_b = df_to_list(df_b)

    result = set([*intervals_a, *intervals_b])
    for ivl_a in intervals_a:
        start_a, end_a, *_ = ivl_a
        for ivl_b in intervals_b:
            start_b, end_b, *_ = ivl_a
            if end_b < start_a:
                continue
            if end_a < start_b:
                break

            if _intervals_intersect(ivl_a, ivl_b):
                if ivl_a in result:
                    result.remove(ivl_a)
                if ivl_b in result:
                    result.remove(ivl_b)

    return (
        pd.DataFrame(result, columns=df_a.columns)
        .set_index("_index")
        .rename_axis(index=None)
    )


@apply_accessor
def diff(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    # Track the index as a column
    df_a = df_a.assign(_index=df_a.index)

    intervals_a = df_to_list(df_a)
    intervals_b = df_to_list(df_b)

    result = set(intervals_a)
    for ivl_a in intervals_a:
        start_a, end_a, *_ = ivl_a
        for ivl_b in intervals_b:
            start_b, end_b, *_ = ivl_a
            if end_b <= start_a:
                continue
            if end_a <= start_b:
                break

            if _intervals_intersect(ivl_a, ivl_b):
                if ivl_a in result:
                    result.remove(ivl_a)

    return (
        pd.DataFrame(result, columns=df_a.columns)
        .set_index("_index")
        .rename_axis(index=None)
    )
