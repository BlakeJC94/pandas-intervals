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


# TODO Implement non-vectorised versions of these operations for tests
class TestIntervalsAccessor:
    def test_format_intervals(self, intervals_df_a, intervals_df_b):
        """Test a `DataFrame` can be formatted by the IntervalsAccessor."""
        ...

    def test_property_duration(self, intervals_df_a):
        """Test a durations of intervals in a `DataFrame` can be dynamically computed by the
        IntervalsAccessor.
        """
        ...

    def test_raise_format_missing_required_column(self, intervals_df_a):
        """Test an exception is raised trying to format a `DataFrame` as intervals with a required
        column missing.
        """
        ...

    def test_empty_intervals_frame(self):
        """Test an empty `DataFrame` can be formatted with correct column types."""
        # Missing columns
        # Missing rows
        ...

    def test_intervals_union(self, intervals_df_a, intervals_df_b):
        """Test an interval union can be computed between two `DataFrame`s of intervals."""
        # Expected
        # Symmetric
        ...

    def test_intervals_union(self, intervals_df_a, intervals_df_b):
        """Test an interval intersection can be computed between two `DataFrame`s of intervals."""
        # Expected
        # Symmetric
        ...
