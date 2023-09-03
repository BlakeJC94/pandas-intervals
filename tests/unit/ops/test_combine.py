import pytest

from pandas_intervals.ops import intervals_combine
from tests.helpers import (
    assert_df_interval_set_equality,
    combine_basic,
    intervals_from_str,
)


@pytest.mark.parametrize(
    "operation",
    [
        combine_basic,
        intervals_combine,
    ],
)
class TestIntervalsCombine:
    @staticmethod
    def check_operation(operation, test_case):
        df_a = intervals_from_str(test_case["a"])
        df_expected_combine_a = intervals_from_str(test_case["combine_a"])
        assert_df_interval_set_equality(df_expected_combine_a, operation(df_a))

    @pytest.mark.parametrize(
        "test_case",
        [
            dict(
                a=[
                    "  |  (----]    (----]    |    (--------------]           ",
                ],
                combine_a=[
                    "  |  (----]    (----]    |    (--------------]           ",
                ],
            ),
            dict(
                a=[
                    "         (--------------](-------------------]           ",
                    "  (------]                                               ",
                ],
                combine_a=[
                    "         (--------------](-------------------]           ",
                    "  (------]                                               ",
                ],
            ),
        ],
    )
    def test_it_doesnt_combine_anything_when_nothing_overlaps(
        self, test_case, operation
    ):
        self.check_operation(operation, test_case)

    @pytest.mark.parametrize(
        "test_case",
        [
            dict(
                a=[
                    "               (----]         (--------------]           ",
                    "                                 (---]    (------]       ",
                ],
                combine_a=[
                    "               (----]         (------------------]       ",
                ],
            ),
            dict(
                a=[
                    "     (----]    (----]         (--------------]           ",
                    " (--]   (---]          (----]    (---]    (------] (---] ",
                ],
                combine_a=[
                    " (--](------]  (----]  (----] (------------------] (---] ",
                ],
            ),
            dict(
                a=[
                    "     (----]                                              ",
                    " (--]   (---]          (----]                      (---] ",
                ],
                combine_a=[
                    " (--](------]          (----]                      (---] ",
                ],
            ),
        ],
    )
    def test_it_combines_overlapping_intervals(self, test_case, operation):
        self.check_operation(operation, test_case)
