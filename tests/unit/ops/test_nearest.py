import pytest
import pandas as pd

from pandas_intervals.ops import intervals_nearest
from pandas_intervals.ops.basic import intervals_nearest as nearest_basic
from tests.helpers import (
    intervals_from_str,
)


@pytest.mark.parametrize("operation", [nearest_basic, intervals_nearest])
class TestIntervalsNearest:
    @staticmethod
    def check_operation(operation, test_case):
        df_a = intervals_from_str(test_case["a"])
        df_b = intervals_from_str(test_case["b"])
        expected = pd.DataFrame({"min_dist": test_case["min_dist"]})
        result = operation(df_a, df_b)
        assert result.iloc[:, 0].tolist() == expected.iloc[:, 0].tolist()

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
