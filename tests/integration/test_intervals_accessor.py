import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytest

import pandas_intervals
from pandas_intervals.utils import df_to_set
from tests.conftest import (
    assert_df_interval_times_equal,
    random_intervals,
)


def union_basic(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    cols = df_a.columns
    intervals_a = df_to_set(df_a)
    intervals_b = df_to_set(df_b)

    result = intervals_a.union(intervals_b)
    return pd.DataFrame(result, columns=cols).sort_values(["start", "end"])


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

    def test_empty(self):
        """Test an empty `DataFrame` can be formatted with correct column types."""
        result = random.choice([pd.DataFrame().ivl(), pd.DataFrame.ivl.empty()])
        assert len(result) == 0
        assert result.columns.tolist()[:2] == pd.DataFrame.ivl.required_cols
        assert all(
            result.dtypes[col] == dtype for col, dtype, _ in pd.DataFrame.ivl.fields
        )

    def test_intervals_contains(self):
        df_a = random_intervals(n_intervals=random.randint(0, 12)).ivl()

        n_selected = random.randint(0, len(df_a) // 2)
        has_other_interval = random.random() < 0.5

        random_mask = np.zeros(len(df_a), dtype=bool)
        random_mask[:n_selected] = True
        np.random.shuffle(random_mask)
        df_b = df_a[random_mask]

        if has_other_interval:
            df_b = pd.concat([df_b, random_intervals(n_intervals=1).ivl()], axis=0)

        result = df_a.ivl.contains(df_b)

        assert result is not has_other_interval

    def test_intervals_pad(self):
        df_a = random_intervals(n_intervals=random.randint(0, 12)).ivl()

        kwargs_0 = [
            ("pad", df_a.ivl.durations.mean() * (random.random() - 0.5)),
        ]
        kwargs_1 = [
            ("left_pad", df_a.ivl.durations.mean() * (random.random() - 0.5)),
            ("right_pad", df_a.ivl.durations.mean() * (random.random() - 0.5)),
        ]
        kwargs = random.choice([kwargs_0, kwargs_1])
        kwargs = dict(random.sample(kwargs, k=random.randint(1, len(kwargs))))

        expected_pad = df_a.copy()
        expected_pad["start"] -= kwargs.get("pad") or kwargs.get("left_pad") or 0
        expected_pad["end"] += kwargs.get("pad") or kwargs.get("right_pad") or 0
        expected_pad = expected_pad[expected_pad["end"] - expected_pad["start"] >= 0]

        df_a_pad = df_a.ivl.pad(**kwargs)

        assert_df_interval_times_equal(
            df_a_pad,
            expected_pad,
        )

    def test_intervals_overlap_and_non_overlap(self):
        """Test an interval overlap can be computed on a `DataFrame` of intervals."""
        df_a = random_intervals(n_intervals=random.randint(0, 12)).ivl()

        df_a_overlap = df_a.ivl.overlap()
        df_a_non_overlap = df_a.ivl.non_overlap()

        expected_overlap = df_a.ivl.basic.overlap()
        expected_non_overlap = df_a.ivl.basic.non_overlap()

        assert_df_interval_times_equal(
            df_a_overlap,
            expected_overlap,
        )
        assert_df_interval_times_equal(
            df_a_non_overlap,
            expected_non_overlap,
        )

        assert_df_interval_times_equal(
            df_a_overlap.ivl.union(df_a_non_overlap),
            df_a,
        )

    def test_intervals_complement(self):
        df_a = random_intervals(n_intervals=random.randint(0, 12)).ivl()

        kwargs = [
            ("left_bound", df_a["start"].min() * (0.5 + random.random())),
            ("right_bound", df_a["end"].max() * (0.5 + random.random())),
        ]
        kwargs = dict(random.sample(kwargs, k=random.randint(0, len(kwargs))))

        df_a_complement = df_a.ivl.complement(**kwargs)
        expected_complement = df_a.ivl.basic.complement(**kwargs)

        assert_df_interval_times_equal(
            df_a_complement,
            expected_complement,
        )

        assert len(df_a.ivl.basic.intersection(df_a_complement)) == 0

    def test_intervals_union(self):
        """Test an interval union can be computed between two `DataFrame`s of intervals."""
        df_a = random_intervals(n_intervals=random.randint(0, 12)).ivl()
        df_b = random_intervals(n_intervals=random.randint(0, 12)).ivl()

        df_a_union_b = df_a.ivl.union(df_b)
        df_b_union_a = df_b.ivl.union(df_a)

        expected = union_basic(df_a, df_b)

        assert_df_interval_times_equal(
            df_a_union_b,
            df_b_union_a,
        )
        assert_df_interval_times_equal(
            df_a_union_b,
            expected,
        )

    def test_intervals_intersection(self):
        """Test an interval intersection can be computed between two `DataFrame`s of intervals."""
        df_a = random_intervals(n_intervals=random.randint(0, 12)).ivl()
        df_b = random_intervals(n_intervals=random.randint(0, 12)).ivl()

        df_a_intersection_b = df_a.ivl.intersection(df_b)
        df_b_intersection_a = df_b.ivl.intersection(df_a)

        expected = df_a.ivl.basic.intersection(df_b)

        assert_df_interval_times_equal(
            df_a_intersection_b,
            df_b_intersection_a,
        )

        assert_df_interval_times_equal(
            df_a_intersection_b,
            expected,
        )

    def test_intervals_combine(self):
        df_a = random_intervals(n_intervals=random.randint(0, 12)).ivl()
        df_combine_a = df_a.ivl.combine()
        expected_combine_a = df_a.ivl.basic.combine()

        assert_df_interval_times_equal(
            df_combine_a,
            expected_combine_a,
        )

    def test_intervals_truncate(self):
        df_a = random_intervals(n_intervals=random.randint(0, 12)).ivl()
        df_b = random_intervals(n_intervals=random.randint(0, 12)).ivl()

        df_a_trunc_b = df_a.ivl.truncate(df_b)
        df_b_trunc_a = df_b.ivl.truncate(df_a)

        expected_a_trunc_b = df_a.ivl.basic.truncate(df_b)
        expected_b_trunc_a = df_b.ivl.basic.truncate(df_a)

        assert_df_interval_times_equal(
            df_a_trunc_b,
            expected_a_trunc_b,
        )
        assert_df_interval_times_equal(
            df_b_trunc_a,
            expected_b_trunc_a,
        )

    def test_intervals_nearest(self):
        df_a = random_intervals(n_intervals=random.randint(0, 12)).ivl()
        df_b = random_intervals(n_intervals=random.randint(0, 12)).ivl()

        result = df_a.ivl.nearest(df_b)
        expected = df_a.ivl.basic.nearest(df_b)

        assert result.iloc[:, 0].tolist() == expected.iloc[:, 0].tolist()
