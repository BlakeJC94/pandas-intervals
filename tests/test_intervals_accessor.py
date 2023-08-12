import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytest

import pandas_intervals
from tests.helpers import (
    assert_df_interval_set_equality,
    random_intervals,
    overlap_basic,
    non_overlap_basic,
    union_basic,
    intersection_basic,
)


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

        assert_df_interval_set_equality(
            df_a_union_b,
            df_b_union_a,
        )
        assert_df_interval_set_equality(
            df_a_union_b,
            expected,
        )

    def test_intervals_overlap(self):
        """Test an interval overlap can be computed on a `DataFrame` of intervals."""
        df_a = random_intervals(n_intervals=random.randint(0, 12))

        df_a_overlap = df_a.ivl.overlap()
        df_a_non_overlap = df_a.ivl.non_overlap()

        expected_overlap = overlap_basic(df_a)
        expected_non_overlap = non_overlap_basic(df_a)

        assert_df_interval_set_equality(
            df_a_overlap,
            expected_overlap,
        )
        assert_df_interval_set_equality(
            df_a_non_overlap,
            expected_non_overlap,
        )

        assert_df_interval_set_equality(
            pd.concat([df_a_overlap, df_a_non_overlap], axis=0),
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
