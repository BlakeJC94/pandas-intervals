import pytest

from tests.conftest import assert_df_interval_times_equal, random_intervals
from tests.acceptance.conftest import benchmark, run


def arrange(n_intervals):
    df_a = random_intervals(n_intervals=n_intervals).ivl()
    return [df_a], {}


@benchmark
def act_0(df_a):
    return df_a.ivl.basic.combine()


@benchmark
def act_1(df_a):
    return df_a.ivl.combine()


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_intervals, expected_ratio",
    [
        (2000, 1),
        (5000, 1.5),
        (200000, 2.5),
    ],
)
def test_combine(n_intervals, expected_ratio, results_record):
    run(
        arrange,
        act_0,
        act_1,
        assert_df_interval_times_equal,
        n_intervals=n_intervals,
        expected_ratio=expected_ratio,
        results_record=results_record,
    )
