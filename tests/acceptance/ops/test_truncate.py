import pytest

from tests.conftest import assert_df_interval_times_equal, random_intervals
from tests.acceptance.conftest import benchmark, run


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_intervals, expected_ratio",
    [
        (1000, 3),
        (2000, 12),
        (5000, 40),
    ],
)
class TestTruncate:
    @staticmethod
    def arrange(n_intervals):
        df_a = random_intervals(n_intervals=n_intervals).ivl()
        df_b = random_intervals(n_intervals=n_intervals).ivl()
        return [df_a, df_b], {}

    @staticmethod
    @benchmark
    def act_0(df_a, df_b):
        return df_a.ivl.basic.truncate(df_b)

    @staticmethod
    @benchmark
    def act_1(df_a, df_b):
        return df_a.ivl.truncate(df_b)

    def test_it_runs_faster_when_vectorised(
        self, n_intervals, expected_ratio, results_record
    ):
        run(
            self.arrange,
            self.act_0,
            self.act_1,
            assert_df_interval_times_equal,
            n_intervals=n_intervals,
            expected_ratio=expected_ratio,
            results_record=results_record,
        )
