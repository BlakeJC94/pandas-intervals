from __future__ import annotations
from typing import Union, List, Mapping, Optional, Set, Any, Tuple, Dict, Callable
from itertools import product

import pandas as pd
import numpy as np


def assert_df_interval_set_equality(df_expected: pd.DataFrame, df_output: pd.DataFrame):
    df_expected = df_expected.astype(dict(start=float, end=float))
    df_output = df_output.astype(dict(start=float, end=float))
    set_expected = df_to_set(df_expected)
    set_output = df_to_set(df_output)
    assert len(set_expected) == len(set_output)
    assert set_expected == set_output
    assert len(df_expected) == len(df_output)


def df_to_set(df: pd.DataFrame) -> Set[List[Any]]:
    return set(df.itertuples(index=False, name=None))


def df_to_list(df: pd.DataFrame) -> Set[List[Any]]:
    return list(df.itertuples(index=False, name=None))


def intervals_from_str(
    intervals_str: Union[str, List[str]],
    tags_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Create an intervals DataFrame from a string input.

    Label starts are represented as '(' and ends are represented as ']' characters. '|' characters
    represent a label start and an end, therefore can be used to represent a zero-duration label or
    start a new label that starts at the previous label end. An alphabetic character after a label
    start is interpreted as the tag (if not given, a default tag '-' is inserted).

    Limitations:
        - Zero-length intervals can't have tags added via string when adjacent to the next label,
        - Overlapping intervals aren't generally supported, use `pd.concat` across multiple outputs,
        - 2-character intervals can't be tagged.

    Examples:
        - Simple intervals (some with tags):
            " (---]  (q----]    (w--] (------]  (--] "
        - Zero duration intervals:
            "    (q----]   |w   |  |     |e   (----] "
        - Tangential intervals:
            "  (q----|-----|w----|e----]     (----]  "
        - Simple, tangential, and zero duration intervals:
            "     (q---]   (w--|e----]   |  |        "

    Args:
        intervals_str: Intervals in string format. Can also accept a list of intervals in string
            format, which can be used for clearer construction label DataFrames of overlapping
            intervals.
        tags_map: An optional mapping from single character tags to longer strings

    Returns:
        DataFrame containing columns `'start', 'end', 'tag'`.
    """
    if isinstance(intervals_str, list):
        return pd.concat([intervals_from_str(s) for s in intervals_str])
    if not isinstance(intervals_str, str):
        raise ValueError("Input must be a string")

    starts, ends, tags = [], [], []
    interval_tags = []
    default_tag = "-"
    i, start, end = 0, None, None
    for i, c in enumerate(intervals_str):
        next_i = min(i + 1, len(intervals_str) - 1)
        if c == "(":  # Start new label
            start = i * 100
            interval_tags = [default_tag] if not intervals_str[next_i].isalnum() else []
        if c == "]":  # End a label
            end = i * 100
            _end_interval(starts, ends, tags, start, end, interval_tags)
            start = None
        if (
            c == "|"
        ):  # Either a zero-duration label, or start a tangential label + end the previous
            if start is not None:
                _end_interval(starts, ends, tags, start, i * 100, interval_tags)
            start = end = i * 100
            interval_tags = [default_tag] if not intervals_str[next_i].isalnum() else []
        elif c.isalnum():  # A tag to add to the label
            interval_tags.append(c)
        elif c == " " and start is not None:  # End a zero-duration label
            _end_interval(starts, ends, tags, start, end, interval_tags)
            start = None
    if start is not None:
        _end_interval(starts, ends, tags, start, i * 100, interval_tags)

    intervals = pd.DataFrame(
        list(zip(starts, ends, tags)), columns=["start", "end", "tag"]
    )
    intervals[["start", "end"]] = intervals[["start", "end"]].astype(float)
    if tags_map:
        intervals["tag"] = intervals["tag"].replace(tags_map)
    return intervals


def _end_interval(starts, ends, tags, start, end, interval_tags):
    for tag in interval_tags:
        starts.append(start)
        ends.append(end)
        tags.append(tag)


def random_intervals(
    n_intervals: int,
    duration_bounds: Optional[Tuple[float, float]] = None,
    gap_bounds: Optional[Tuple[float, float]] = None,
):
    if n_intervals == 0:
        return pd.DataFrame.ivl.empty()

    if duration_bounds is None:
        duration_bounds = (10, 60)
    if gap_bounds is None:
        gap_bounds = (-10, 10)

    min_duration, max_duration = duration_bounds
    durations = min_duration + np.random.rand(n_intervals) * (
        max_duration - min_duration
    )
    durations = np.round(durations, 2)

    min_gap, max_gap = gap_bounds
    gaps = min_gap + np.random.rand(n_intervals) * (max_gap - min_gap)
    gaps = np.round(gaps, 2)

    offset = np.random.rand() * np.mean(
        np.abs(np.concatenate([duration_bounds, gap_bounds]))
    )
    starts = np.round(offset, 2) + np.insert(np.cumsum(gaps + durations)[:-1], 0, 0)
    ends = starts + durations
    return pd.DataFrame(np.stack([starts, ends], axis=1)).ivl()


def _overlap_mask_basic(df_a: pd.DataFrame) -> np.ndarray:
    intervals_a = df_to_list(df_a)

    mask = []
    for start_a, end_a in intervals_a:
        overlap = False
        for start_b, end_b in intervals_a:
            if (
                (start_a < start_b < end_a)
                or (start_a < end_b < end_a)
                or (start_b < start_a < end_b)
                or (start_b < end_a < end_b)
            ):
                overlap = True
                break
        mask.append(overlap)

    return np.array(mask)


def overlap_basic(df_a: pd.DataFrame) -> pd.DataFrame:
    if df_a.empty:
        return df_a
    return df_a.loc[_overlap_mask_basic(df_a)]


def non_overlap_basic(df_a: pd.DataFrame) -> pd.DataFrame:
    if df_a.empty:
        return df_a
    return df_a.loc[~_overlap_mask_basic(df_a)]


def union_basic(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    cols = df_a.columns
    intervals_a = df_to_set(df_a)
    intervals_b = df_to_set(df_b)

    result = intervals_a.union(intervals_b)
    return pd.DataFrame(result, columns=cols).sort_values(["start", "end"])


def intersection_basic(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    cols = df_a.columns
    intervals_a = df_to_list(df_a)
    intervals_b = df_to_list(df_b)

    result = set()
    for ivl_a, ivl_b in product(intervals_a, intervals_b):
        start_a, end_a, *_ = ivl_a
        start_b, end_b, *_ = ivl_b
        if (
            (start_a < start_b < end_a)
            or (start_a < end_b < end_a)
            or (start_b < start_a < end_b)
            or (start_b < end_a < end_b)
        ):
            result.add(ivl_a)
            result.add(ivl_b)

    return pd.DataFrame(result, columns=cols).sort_values(["start", "end"])

def combine_basic(
    df: pd.DataFrame,
    aggregations: Optional[Dict[str, Union[str, Callable]]] = None,
):
    if df.empty:
        return df

    if aggregations is None:
        aggregations = {"start": "min", "end": "max"}

    aggregations.update({c: "first" for c in df.columns if c not in aggregations})
    df_sorted = df.sort_values("start")

    # Loop over labels and compare each to the previous label to find labels to combine
    group_inds = []
    ind, interval_end_time = 0, 0
    for start, end in df_sorted[["start", "end"]].values:
        # If interval is within previous label, combine them
        if start <= interval_end_time:
            interval_end_time = max(interval_end_time, end)
            group_inds.append(ind)
        # If not, start a new interval
        else:
            interval_end_time = end
            ind += 1
            group_inds.append(ind)

    return df_sorted.groupby(group_inds).agg(aggregations)

# TODO finish this
def complement_basic(
    df_a: pd.DataFrame,
    left_bound: Optional[float] = None,
    right_bound: Optional[float] = None,
):
    if left_bound is None:
        left_bound = df_a['start'].min()
    if right_bound is None:
        right_bound = df_a['end'].max()

    ...
