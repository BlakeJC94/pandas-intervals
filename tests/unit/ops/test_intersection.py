import pytest

from pandas_intervals.ops import intersection
from pandas_intervals.ops.basic import intersection as intersection_basic
from pandas_intervals.ops.intersection import _get_mask_no_ref_overlap
from tests.conftest import (
    assert_df_interval_times_equal,
    intervals_from_str,
)


def test_get_mask_no_ref_overlap_tangential_intervals():
    df_a = intervals_from_str("  (-----]       ")
    df_b = intervals_from_str("        (-----] ")
    assert not _get_mask_no_ref_overlap(df_a, df_b).any()
    assert not _get_mask_no_ref_overlap(df_b, df_a).any()

@pytest.mark.parametrize("operation", [intersection_basic, intersection])
class TestIntersection:
    @staticmethod
    def check_operation(operation, test_case):
        df_a = intervals_from_str(test_case["a"])
        df_b = intervals_from_str(test_case["b"])
        df_expected = intervals_from_str(test_case["a_intersection_b"])
        assert_df_interval_times_equal(df_expected, operation(df_a, df_b))
        assert_df_interval_times_equal(df_expected, operation(df_b, df_a))

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

@pytest.mark.parametrize("operation", [
    symdiff_basic,
    symdiff
])
class TestSymdiff:
    @staticmethod
    def check_operation(operation, test_case):
        df_a = intervals_from_str(test_case["a"])
        df_b = intervals_from_str(test_case["b"])
        df_expected = intervals_from_str(test_case["expected"])
        assert_df_interval_times_equal(df_expected, operation(df_a, df_b), df_a, df_b,
                                       plot_on_error=True)
        assert_df_interval_times_equal(df_expected, operation(df_b, df_a), df_a, df_b,
                                       plot_on_error=True)

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
                expected=[
                    "               (----]                                    ",
                    " (--]                  (----]                      (---] ",
                ],
            ),
        ],
    )
    def test_it_calculates_symdiff_args_no_overlap_within(self, test_case, operation):
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
                expected=[
                    "     (----]    (----]         (--------------]           ",
                    " (--]   (---]          (----]    (---]    (------] (---] ",
                ],
            ),
        ],
    )
    def test_it_returns_everything_when_args_mutually_exclusive(self, test_case, operation):
        self.check_operation(operation, test_case)
