import pytest

from tests.conftest import random_intervals
from tests.acceptance.conftest import benchmark, run


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_intervals, expected_ratio",
    [
        (1000, 3),
        (2000, 12),
        (4000, 40),
    ],
)
class TestNearest:
    @staticmethod
    def arrange(n_intervals):
        df_a = random_intervals(n_intervals=n_intervals).ivl()
        df_b = random_intervals(n_intervals=n_intervals).ivl()
        return [df_a, df_b], {}

    @staticmethod
    @benchmark
    def act_0(df_a, df_b):
        return df_a.ivl.basic.nearest(df_b)

    @staticmethod
    @benchmark
    def act_1(df_a, df_b):
        return df_a.ivl.nearest(df_b)

    @staticmethod
    def check(df_a, df_b, *_):
        assert (df_a["min_dist"] == df_b["min_dist"]).all()

    def test_it_runs_faster_when_vectorised(
        self, n_intervals, expected_ratio, results_record
    ):
        run(
            self.arrange,
            self.act_0,
            self.act_1,
            self.check,
            n_intervals=n_intervals,
            expected_ratio=expected_ratio,
            results_record=results_record,
        )
