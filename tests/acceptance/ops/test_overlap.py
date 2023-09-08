import pytest

from tests.conftest import assert_df_interval_times_equal, random_intervals
from tests.acceptance.conftest import benchmark, run


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_intervals, expected_ratio",
    [
        (2000, 1),
        (4000, 1.5),
        (8000, 3),
    ],
)
class TestOverlap:
    @staticmethod
    def arrange(n_intervals):
        df_a = random_intervals(n_intervals=n_intervals).ivl()
        return [df_a], {}

    @staticmethod
    @benchmark
    def act_0(df_a):
        return df_a.ivl.basic.overlap()

    @staticmethod
    @benchmark
    def act_1(df_a):
        return df_a.ivl.overlap()

    def test_complement(self, n_intervals, expected_ratio, results_record):
        run(
            self.arrange,
            self.act_0,
            self.act_1,
            assert_df_interval_times_equal,
            n_intervals=n_intervals,
            expected_ratio=expected_ratio,
            results_record=results_record,
        )
