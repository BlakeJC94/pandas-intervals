import pytest

from pandas_intervals.ops import overlap, non_overlap
from pandas_intervals.ops.basic import (
    overlap as overlap_basic,
    non_overlap as non_overlap_basic,
)
from tests.conftest import (
    assert_df_interval_times_equal,
    intervals_from_str,
)


@pytest.mark.parametrize(
    "operation, operation_non",
    [
        (overlap_basic, non_overlap_basic),
        (overlap, non_overlap),
    ],
)
class TestIntervalsOverlap:
    @staticmethod
    def check_operations(operations, test_case):
        df_a = intervals_from_str(test_case["a"])
        df_expected_overlap_a = intervals_from_str(test_case["overlap_a"])
        df_expected_non_overlap_a = intervals_from_str(test_case["non_overlap_a"])
        assert_df_interval_times_equal(df_expected_overlap_a, operations[0](df_a))
        assert_df_interval_times_equal(df_expected_non_overlap_a, operations[1](df_a))

    @pytest.mark.parametrize(
        "test_case",
        [
            dict(
                a=[
                    "     (----]    (----]   |     (--------------]           ",
                ],
                overlap_a=[
                    "                                                         ",
                ],
                non_overlap_a=[
                    "     (----]    (----]   |     (--------------]           ",
                ],
            ),
        ],
    )
    def test_it_returns_nothing_when_arg_has_no_overlaps(
        self, test_case, operation, operation_non
    ):
        self.check_operations((operation, operation_non), test_case)

    @pytest.mark.parametrize(
        "test_case",
        [
            dict(
                a=[
                    "               (----]         (--------------]           ",
                    "                                 (---]    (------]       ",
                ],
                overlap_a=[
                    "                              (--------------]           ",
                    "                                 (---]    (------]       ",
                ],
                non_overlap_a=[
                    "               (----]                                    ",
                ],
            ),
            dict(
                a=[
                    "     (----]                                              ",
                    " (--]   (---]          (----]                      (---] ",
                ],
                overlap_a=[
                    "     (----]                                              ",
                    "        (---]                                            ",
                ],
                non_overlap_a=[
                    " (--]                  (----]                      (---] ",
                ],
            ),
        ],
    )
    def test_it_returns_overlaps_when_arg_has_overlaps(
        self, test_case, operation, operation_non
    ):
        self.check_operations((operation, operation_non), test_case)

    @pytest.mark.parametrize(
        "test_case",
        [
            dict(
                a=[
                    "       (---]   (----]                                    ",
                    "  (----]            (---]                                ",
                ],
                overlap_a=[
                    "                                                         ",
                ],
                non_overlap_a=[
                    "       (---]   (----]                                    ",
                    "  (----]            (---]                                ",
                ],
            ),
            dict(  # TODO Revisit right point overlap logic
                a=[
                    "       (---]   (----]      |                             ",
                    "       |            |      |                             ",
                ],
                overlap_a=[
                    "                                                         ",
                ],
                non_overlap_a=[
                    "       (---]   (----]      |                             ",
                    "       |            |      |                             ",
                ],
            ),
        ],
    )
    def test_it_handles_edges(self, test_case, operation, operation_non):
        self.check_operations((operation, operation_non), test_case)
