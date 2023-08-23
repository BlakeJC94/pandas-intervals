import pytest

from pandas_intervals.ops import intervals_difference
from tests.helpers import (
    assert_df_interval_set_equality,
    difference_basic,
    intervals_from_str,
)


@pytest.mark.parametrize("operation", [difference_basic, intervals_difference])
class TestIntervalsDifference:
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
    def test_it_calculates_results_for_args_no_overlap_within(
        self, test_case, operation
    ):
        df_a = intervals_from_str(test_case["a"])
        df_b = intervals_from_str(test_case["b"])
        df_expected_a_diff_b = intervals_from_str(test_case["a_diff_b"])
        assert_df_interval_set_equality(df_expected_a_diff_b, operation(df_a, df_b))

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
                    " (--](------]          (----]                      (---] ",
                ],
            ),
        ],
    )
    def test_it_calculates_results_for_args_overlap_within(self, test_case, operation):
        df_a = intervals_from_str(test_case["a"])
        df_b = intervals_from_str(test_case["b"])
        df_expected_a_diff_b = intervals_from_str(test_case["a_diff_b"])
        assert_df_interval_set_equality(df_expected_a_diff_b, operation(df_a, df_b))

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
        df_a = intervals_from_str(test_case["a"])
        df_b = intervals_from_str(test_case["b"])
        df_expected_a_diff_b = intervals_from_str(test_case["a_diff_b"])
        assert_df_interval_set_equality(df_expected_a_diff_b, operation(df_a, df_b))

    @pytest.mark.parametrize(
        "test_case",
        [
            # dict(
            #     a=[
            #         "  (----]    (----]    (----]        (-----] ",
            #     ],
            #     b=[
            #         "     |      |              |                ",
            #     ],
            #     a_diff_b=[
            #         "  (--]      (----]    (----]        (-----] ",
            #         "     (-]                                    ",
            #     ],
            # ),
            # dict(  # TODO
            #     a=[
            #         "     |      |              |     |          ",
            #     ],
            #     b=[
            #         "  (----]    (----]    (----]        (-----] ",
            #     ],
            #     a_diff_b=[
            #         "            |              |     |          ",
            #     ],
            # ),
        ],
    )
    def test_it_accepts_zero_duration_inputs(self, test_case, operation):
        df_a = intervals_from_str(test_case["a"])
        df_b = intervals_from_str(test_case["b"])
        df_expected_a_diff_b = intervals_from_str(test_case["a_diff_b"])
        assert_df_interval_set_equality(df_expected_a_diff_b, operation(df_a, df_b))

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
        df_a = intervals_from_str(test_case["a"])
        df_b = intervals_from_str(test_case["b"])
        df_expected_a_diff_b = intervals_from_str(test_case["a_diff_b"])
        assert_df_interval_set_equality(df_expected_a_diff_b, operation(df_a, df_b))
