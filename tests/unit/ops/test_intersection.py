import pytest

from pandas_intervals.ops import intersection
from pandas_intervals.ops.basic import intersection as intersection_basic
from tests.helpers import (
    assert_df_interval_set_equality,
    intervals_from_str,
)


@pytest.mark.parametrize("operation", [intersection_basic, intersection])
class TestIntervalsIntersection:
    @staticmethod
    def check_operation(operation, test_case):
        df_a = intervals_from_str(test_case["a"])
        df_b = intervals_from_str(test_case["b"])
        df_expected = intervals_from_str(test_case["a_intersection_b"])
        assert_df_interval_set_equality(df_expected, operation(df_a, df_b))
        assert_df_interval_set_equality(df_expected, operation(df_b, df_a))

    @pytest.mark.parametrize(
        "test_case",
        [
            dict(
                a=[
                    "     (----]    (----]         (--------------]           ",
                ],
                b=[
                    " (--]   (---]          (----]    (---]    (------] (---] ",
                ],
                a_intersection_b=[
                    "     (----]                   (--------------]           ",
                    "        (---]                    (---]    (------]       ",
                ],
            ),
        ],
    )
    def test_it_calculates_intersection_args_no_overlap_within(self, test_case, operation):
        self.check_operation(operation, test_case)

    @pytest.mark.parametrize(
        "test_case",
        [
            dict(
                a=[
                    "     (----]                                              ",
                    " (--]   (---]          (----]                      (---] ",
                ],
                b=[
                    "               (----]         (--------------]           ",
                    "                                 (---]    (------]       ",
                ],
                a_intersection_b=[
                    "                                                         ",
                ],
            ),
        ],
    )
    def test_it_returns_nothing_when_args_mutually_exclusive(self, test_case, operation):
        self.check_operation(operation, test_case)
