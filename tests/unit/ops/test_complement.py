import pytest

from pandas_intervals.ops import complement
from pandas_intervals.ops.basic import complement as complement_basic
from tests.conftest import (
    assert_df_interval_times_equal,
    intervals_from_str,
)


@pytest.mark.parametrize("operation", [complement_basic, complement])
class TestIntervalsComplement:
    @staticmethod
    def check_operation(operation, test_case):
        df_a = intervals_from_str(test_case["a"])
        df_expected = intervals_from_str(test_case["complement_a"])
        assert_df_interval_times_equal(df_expected, operation(df_a))

    @pytest.mark.parametrize(
        "test_case",
        [
            dict(
                a=[
                    "     (----]    (----]         (--------------]           ",
                ],
                complement_a=[
                    "          (----]    (---------]                          ",
                ],
            ),
            dict(
                a=[
                    " (--]   (---]          (----]    (---]    (------] (---] ",
                ],
                complement_a=[
                    "    (---]   (----------]    (----]   (----]      (-]     ",
                ],
            ),
        ],
    )
    def test_it_returns_intervals_made_from_spaces_between_input(
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
                complement_a=[
                    "                    (---------]                          ",
                ],
            ),
            dict(
                a=[
                    "     (----]    (----]         (--------------]           ",
                    " (--]   (---]          (----]    (---]    (------] (---] ",
                ],
                complement_a=[
                    "    (]      (--]    (--]    (-]                  (-]     ",
                ],
            ),
        ],
    )
    def test_it_returns_gaps_for_overlapping_inputs(self, test_case, operation):
        self.check_operation(operation, test_case)
