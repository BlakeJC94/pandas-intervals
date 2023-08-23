import pytest

from pandas_intervals.ops import intervals_nearest
from tests.helpers import (
    nearest_basic,
    intervals_from_str,
)


@pytest.mark.parametrize("operation", [nearest_basic, intervals_nearest])
class TestIntervalsNearest:
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
        df_a = intervals_from_str(test_case["a"])
        df_b = intervals_from_str(test_case["b"])
        expected = test_case["min_dist"]
        result = operation(df_a, df_b)
        assert result["min_dist"].tolist() == expected
