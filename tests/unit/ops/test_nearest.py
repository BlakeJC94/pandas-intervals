import pytest
import pandas as pd

from pandas_intervals.ops import nearest
from pandas_intervals.ops.basic import nearest as nearest_basic
from tests.conftest import (
    intervals_from_str,
)


@pytest.mark.parametrize("operation", [nearest_basic, nearest])
class TestIntervalsNearest:
    @staticmethod
    def check_operation(operation, test_case):
        df_a = intervals_from_str(test_case["a"])
        df_b = intervals_from_str(test_case["b"])
        expected = pd.Series(test_case["min_dist"])
        result = operation(df_a, df_b)
        assert result.tolist() == expected.tolist()

    @pytest.mark.parametrize(
        "test_case",
        [
            dict(
                a=[
                    " (----]         (-----]    ",
                ],
                b=[
                    "        (--]           (-] ",
                ],
                min_dist=[200, 100],
            ),
        ],
    )
    def test_it_calculates_simple_distances(self, test_case, operation):
        self.check_operation(operation, test_case)
