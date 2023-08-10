from __future__ import annotations
from typing import Union, List, Mapping, Optional

import pandas as pd
import numpy as np


def labels_from_str(
    labels_str: Union[str, List[str]],
    tags_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Create an labels DataFrame from a string input.

    Label starts are represented as '(' and ends are represented as ')' characters. '|' characters
    represent a label start and an end, therefore can be used to represent a zero-duration label or
    start a new label that starts at the previous label end. An alphabetic character after a label
    start is interpreted as the tag (if not given, a default tag '-' is inserted).

    Limitations:
        - Zero-length labels can't have tags added via string when adjacent to the next label,
        - Overlapping labels aren't generally supported, use `pd.concat` across multiple outputs,
        - 2-character labels can't be tagged.

    Examples:
        - Simple labels (some with tags):
            " (---)  (q----)    (w--) (------)  (--) "
        - Zero duration labels:
            "    (q----)   |w   |  |     |e   (----) "
        - Tangential labels:
            "  (q----|-----|w----|e----)     (----)  "
        - Simple, tangential, and zero duration labels:
            "     (q---)   (w--|e----)   |  |        "

    Args:
        labels_str: Labels in string format. Can also accept a list of labels in string format,
            which can be used for clearer construction label DataFrames of overlapping labels.
        tags_map: An optional mapping from single character tags to longer strings

    Returns:
        DataFrame containing columns `'start', 'end', 'tag'`.
    """
    if isinstance(labels_str, list):
        return pd.concat([labels_from_str(s) for s in labels_str])
    if not isinstance(labels_str, str):
        raise ValueError("Input must be a string")

    starts, ends, tags = [], [], []
    default_tag = "-"
    i, start = 0, None
    for i, c in enumerate(labels_str):
        next_i = min(i + 1, len(labels_str) - 1)
        if c == "(":  # Start new label
            start = i * 100
            label_tags = [default_tag] if not labels_str[next_i].isalnum() else []
        if c == ")":  # End a label
            end = i * 100
            _end_label(starts, ends, tags, start, end, label_tags)
            start = None
        if (
            c == "|"
        ):  # Either a zero-duration label, or start a tangential label + end the previous
            if start is not None:
                _end_label(starts, ends, tags, start, i * 100, label_tags)
            start = end = i * 100
            label_tags = [default_tag] if not labels_str[next_i].isalnum() else []
        elif c.isalnum():  # A tag to add to the label
            label_tags.append(c)
        elif c == " " and start is not None:  # End a zero-duration label
            _end_label(starts, ends, tags, start, end, label_tags)
            start = None
    if start is not None:
        _end_label(starts, ends, tags, start, i * 100, label_tags)

    labels = pd.DataFrame(
        list(zip(starts, ends, tags)), columns=["start", "end", "tag"]
    )
    labels[["start", "end"]] = labels[["start", "end"]].astype(float)
    if tags_map:
        labels["tag"] = labels["tag"].replace(tags_map)
    return labels


def _end_label(starts, ends, tags, start, end, label_tags):
    for tag in label_tags:
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
    intervals_a = df_a.iloc[:, :2].values

    mask = []
    for start_a, end_a in intervals_a:
        overlap = False
        for start_b, end_b in intervals_a:
            if (
                (start_b < start_a < end_b < end_a)
                or (start_a < start_b < end_a < end_b)
                or ((start_a < start_b) and (end_b < end_a))
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
    intervals_a = df_a.iloc[:, :2].values
    intervals_b = df_b.iloc[:, :2].values

    result = np.concatenate([intervals_a, intervals_b])
    result = np.unique(result, axis=0)
    return (
        pd.DataFrame(result, columns=["start", "end"])
        .sort_values(["start", "end"])
        .astype(float)
    )


def intersection_basic(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    intervals_a = df_a.iloc[:, :2].values
    intervals_b = df_b.iloc[:, :2].values

    idxs_a, idxs_b = [], []
    for idx_a, (start_a, end_a) in enumerate(intervals_a):
        for idx_b, (start_b, end_b) in enumerate(intervals_b):
            if (
                (start_b < start_a < end_b < end_a)
                or (start_a < start_b < end_a < end_b)
                or ((start_a < start_b) and (end_b < end_a))
            ):
                idxs_a.append(idx_a)
                idxs_b.append(idx_b)

    result = np.concatenate([intervals_a[idxs_a], intervals_b[idxs_b]])
    return (
        pd.DataFrame(result, columns=["start", "end"])
        .sort_values(["start", "end"])
        .astype(float)
    )
