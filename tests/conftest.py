import random
from typing import Union, List, Mapping, Optional, Any, Tuple

import numpy as np
import pandas as pd
import pytest

from pandas_intervals.vis import _plot_interval_groups as plt


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def assert_df_interval_times_equal(
    df_expected: pd.DataFrame,
    df_output: pd.DataFrame,
    *other_dfs,
    plot_on_error: bool = False,
):
    df_expected = df_expected[["start", "end"]].sort_values(["start", "end"])
    df_output = df_output[["start", "end"]].sort_values(["start", "end"])

    times_expected = df_expected.to_numpy()
    times_output = df_output.to_numpy()
    try:
        assert len(times_output) == len(
            times_expected
        ), f"n_expected = {len(times_expected)}, n_output = {len(times_output)}"
        assert (times_output == times_expected).all()
    except Exception as err:
        difflen = len(times_expected) - len(times_output)
        times_expected = np.concatenate(
            [
                times_expected,
                np.zeros((max(-difflen, 0), 2)),
            ],
            axis=0,
        )
        times_output = np.concatenate(
            [
                times_output,
                np.zeros((max(difflen, 0), 2)),
            ],
            axis=0,
        )
        idx_row = np.min(np.where(times_expected != times_output)[0])

        idx_start = max(idx_row - 2, 0)
        win_start = min(
            df_expected.iloc[idx_start]["start"], df_output.iloc[idx_start]["start"]
        )

        idx_end = min(idx_row + 3, len(df_expected) - 1, len(df_output) - 1)
        win_end = max(df_expected.iloc[idx_end]["end"], df_output.iloc[idx_end]["end"])

        foo = df_expected[
            (df_expected["end"] > win_start) & (df_expected["start"] < win_end)
        ]
        bar = df_output[(df_output["end"] > win_start) & (df_output["start"] < win_end)]
        other_dfs = [
            df[(df["end"] > win_start) & (df["start"] < win_end)] for df in other_dfs
        ]

        if plot_on_error:
            plt(*other_dfs, foo, bar)

        raise err


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
    if all(t == default_tag for t in tags):
        intervals = intervals.drop(columns=["tag"])
    elif tags_map:
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
    random_fields: Optional[List[Tuple[str, List[Any]]]] = None,
):
    if n_intervals == 0:
        return pd.DataFrame()

    random_fields = random_fields or []

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

    data = []
    for start, end in zip(starts, ends):
        md = [random.choice(v) for _, v in random_fields]
        data.append((start, end, *md))

    return pd.DataFrame(data, columns=["start", "end", *[k for k, _ in random_fields]])
