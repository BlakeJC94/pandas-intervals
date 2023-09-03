import pytest

from pandas_intervals.ops import intervals_truncate
from pandas_intervals.vis import plot_interval_groups as plt
from tests.helpers import (
    assert_df_interval_set_equality,
    truncate_basic,
    intervals_from_str,
)


@pytest.mark.parametrize(
    "operation",
    [
        truncate_basic,
        intervals_truncate,
    ],
)
class TestIntervalsDifference:
    @staticmethod
    def check_operation(operation, test_case):
        df_a = intervals_from_str(test_case["a"])
        df_b = intervals_from_str(test_case["b"])
        df_expected = intervals_from_str(test_case["a_diff_b"])
        df_output = operation(df_a, df_b)
        assert_df_interval_set_equality(df_expected, df_output)

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
                a_diff_b=[
                    "     (--]      (----]         (--]   (----]              ",
                ],
            ),
        ],
    )
    def test_it_calculates_results_for_args_no_overlap_within(
        self, test_case, operation
    ):
        self.check_operation(operation, test_case)

    @pytest.mark.parametrize(
        "test_case",
        [
            dict(
                a=[
                    "  (----]    (----]    (----]        (-----] ",
                ],
                b=[
                    " (--]     (--]                              ",
                ],
                a_diff_b=[
                    "    (--]     (---]    (----]        (-----] ",
                ],
            ),
            dict(
                a=[
                    "  (----]    (----]    (----]        (-----] ",
                ],
                b=[
                    "     (--]     (--]                          ",
                ],
                a_diff_b=[
                    "  (--]      (-]       (----]        (-----] ",
                ],
            ),
            dict(
                a=[
                    "  (----]    (----]    (----]        (-----] ",
                ],
                b=[
                    " (--]         (--]                          ",
                ],
                a_diff_b=[
                    "    (--]    (-]       (----]        (-----] ",
                ],
            ),
        ],
    )
    def test_it_trims_subtracts_one_side(self, test_case, operation):
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
                a_diff_b=[
                    "     (----]                                              ",
                    " (--]   (---]          (----]                      (---] ",
                ],
            ),
            dict(
                a=[
                    "     (----]                                              ",
                    " (--]   (---]          (----]                      (---] ",
                ],
                b=[
                    "         (----------]         (--------------]           ",
                    "                                 (---]    (------]       ",
                ],
                a_diff_b=[
                    "     (---]                                               ",
                    " (--]   (]             (----]                      (---] ",
                ],
            ),
        ],
    )
    def test_it_calculates_results_for_args_overlap_within(self, test_case, operation):
        self.check_operation(operation, test_case)

    @pytest.mark.parametrize(
        "test_case",
        [
            dict(
                a=[
                    "  (----]    (----]    (----]        (-----] ",
                ],
                b=[
                    "        (--]      (--]       (---]          ",
                ],
                a_diff_b=[
                    "  (----]    (----]    (----]        (-----] ",
                ],
            ),
            dict(
                a=[
                    "                (----]    (----]    (----]        (-----]                ",
                ],
                b=[
                    " (--] (--] (--]                                           (--] (--] (--] ",
                ],
                a_diff_b=[
                    "                (----]    (----]    (----]        (-----]                ",
                ],
            ),
        ],
    )
    def test_it_returns_unmodified_inputs_when_args_dont_overlap(
        self, test_case, operation
    ):
        self.check_operation(operation, test_case)

    @pytest.mark.parametrize(
        "test_case",
        [
            dict(
                a=[
                    "  (----]    (----]    (----]        (-----] ",
                ],
                b=[
                    "     |      |              |                ",
                ],
                a_diff_b=[
                    "  (--]      (----]    (----]        (-----] ",
                    "     (-]                                    ",
                ],
            ),
            dict(
                a=[
                    "     |      |              |     |          ",
                ],
                b=[
                    "  (----]    (----]    (----]        (-----] ",
                ],
                a_diff_b=[
                    "            |                    |          ",
                ],
            ),
        ],
    )
    def test_it_accepts_zero_duration_inputs(self, test_case, operation):
        self.check_operation(operation, test_case)

    @pytest.mark.parametrize(
        "test_case",
        [
            dict(
                a=[
                    "  (----]    (----]    (----]        (-----] ",
                ],
                b=[
                    " (------]  (--]                             ",
                ],
                a_diff_b=[
                    "              (--]    (----]        (-----] ",
                ],
            ),
            dict(
                a=[
                    "  (----]    (----]    (----]        (-----] ",
                ],
                b=[
                    "     (--] (--------]                        ",
                ],
                a_diff_b=[
                    "  (--]                (----]        (-----] ",
                ],
            ),
            dict(
                a=[
                    "  (----]    (----]    (----]        (-----] ",
                ],
                b=[
                    " (-------] (--------]                       ",
                ],
                a_diff_b=[
                    "                      (----]        (-----] ",
                ],
            ),
            dict(
                a=[
                    " (------------------](-------------------]    ",
                ],
                b=[
                    "   (----]    (----]    (----]        (-----]  ",
                ],
                a_diff_b=[
                    " (-]    (----]    (-](-]    (--------]        ",
                ],
            ),
        ],
    )
    def test_it_drops_totally_overlapped_intervals(self, test_case, operation):
        self.check_operation(operation, test_case)
