import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytest

import pandas_intervals


@pytest.fixture
def intervals_df_a():
    """A DataFrame with correct column names and integer types."""
    return pd.DataFrame(
        [
            [0, 100],
            [0, 100],
            [200, 350],
            [1000, 2000],
        ],
        columns=["start", "end"],
    )


@pytest.fixture
def intervals_df_b():
    """A DataFrame without column names and integer types."""
    return pd.DataFrame(
        [
            [10, 40],
            [80, 120],
            [230, 280],
            [330, 370],
            [420, 490],
            [510, 550],
        ],
    )


class TestIntervalsAccessor:
    def test_format_intervals(self, intervals_df_a):
        """Test a `DataFrame` can be formatted by the IntervalsAccessor."""
        data = random.choice([None, intervals_df_a.values])
        cols = random.choice([None, ["start", "end"]])

        df = pd.DataFrame(data, columns=cols)

        result = df.ivl()
        assert result.columns.tolist() == ["start", "end"]
        assert result.dtypes.tolist() == [float, float]

    def test_property_duration(self, intervals_df_a):
        """Test a durations of intervals in a `DataFrame` can be dynamically computed by the
        IntervalsAccessor.
        """
        expected_durations = (intervals_df_a["end"] - intervals_df_a["start"]).astype(
            float
        )
        assert expected_durations.equals(intervals_df_a.ivl.durations)

    def test_raise_format_missing_required_column(self, intervals_df_a):
        """Test an exception is raised trying to format a `DataFrame` as intervals with a required
        column missing.
        """
        drop_col = random.choice(intervals_df_a.ivl.required_cols)
        partial_intervlas_df = intervals_df_a.drop(drop_col, axis=1)
        with pytest.raises(ValueError):
            partial_intervlas_df.ivl()

    def test_empty_intervals_frame(self):
        """Test an empty `DataFrame` can be formatted with correct column types."""
        result = pd.DataFrame.ivl.empty()
        assert len(result) == 0
        assert result.columns.tolist()[:2] == pd.DataFrame.ivl.required_cols
        assert all(
            result.dtypes[col] == dtype for col, dtype, _ in pd.DataFrame.ivl.fields
        )

    def test_intervals_union(self):
        """Test an interval union can be computed between two `DataFrame`s of intervals."""
        df_a = random_intervals(n_intervals=random.randint(0, 12))
        df_b = random_intervals(n_intervals=random.randint(0, 12))

        df_a_union_b = df_a.ivl.union(df_b)
        df_b_union_a = df_b.ivl.union(df_a)

        expected = union_basic(df_a, df_b)

        pd.testing.assert_frame_equal(
            df_a_union_b.reset_index(drop=True),
            df_b_union_a.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            df_a_union_b.reset_index(drop=True),
            expected.reset_index(drop=True),
        )

    def test_intervals_overlap(self):
        """Test an interval overlap can be computed on a `DataFrame` of intervals."""
        df_a = random_intervals(n_intervals=random.randint(0, 12))

        df_a_overlap = df_a.ivl.overlap()
        df_a_non_overlap = df_a.ivl.non_overlap()

        expected_overlap = overlap_basic(df_a)
        expected_non_overlap = non_overlap_basic(df_a)

        pd.testing.assert_frame_equal(
            df_a_overlap,
            expected_overlap,
        )
        pd.testing.assert_frame_equal(
            df_a_non_overlap,
            expected_non_overlap,
        )

        result = pd.concat([df_a_overlap, df_a_non_overlap], axis=0)
        pd.testing.assert_frame_equal(
            result.sort_values("start"),
            df_a,
        )

    @pytest.mark.skip
    def test_intervals_intersection(self):
        """Test an interval intersection can be computed between two `DataFrame`s of intervals."""
        # TODO Re-enable prop-based test once debugged
        # df_a = random_intervals(n_intervals=random.randint(0, 12))
        # df_b = random_intervals(n_intervals=random.randint(0, 12))

        df_a = pd.DataFrame(
            [
                [20.88, 56.74],
                [57.61, 108.3],
                [110.37, 123.21],
                [127.73, 180.88],
                [188.3, 240.22],
                [233.43, 274.27],
                [279.44, 326.84],
                [325.86, 338.04],
                [331.45, 365.37],
                [365.46, 386.41],
            ]
        ).ivl()
        df_b = pd.DataFrame.ivl.empty()

        df_a_intersection_b = df_a.ivl.intersection(df_b)
        df_b_intersection_a = df_b.ivl.intersection(df_a)

        expected = intersection_basic(df_a, df_b)

        pd.testing.assert_frame_equal(
            df_a_intersection_b.reset_index(drop=True),
            df_b_intersection_a.reset_index(drop=True),
        )
        breakpoint()
        pd.testing.assert_frame_equal(
            df_a_intersection_b.reset_index(drop=True),
            expected.reset_index(drop=True),
        )
        ...


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


def non_overlap_basic(df_a: pd.DataFrame) -> pd.DataFrame:
    if df_a.empty:
        return df_a
    mask = _overlap_mask_basic(df_a)
    return df_a.loc[~mask]


def overlap_basic(df_a: pd.DataFrame) -> pd.DataFrame:
    if df_a.empty:
        return df_a
    mask = _overlap_mask_basic(df_a)
    return df_a.loc[mask]


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
