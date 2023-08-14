import pytest
import pandas as pd

import pandas_intervals
from tests.helpers import (
    assert_df_interval_set_equality,
    combine_basic,
    intervals_from_str,
    overlap_basic,
    non_overlap_basic,
    union_basic,
    intersection_basic,
    complement_basic,
    diff_basic,
)

cases = [
    dict(
        A=[
            "     (----]    (----]         (--------------]           ",
        ],
        B=[
            " (--]   (---]          (----]    (---]    (------] (---] ",
        ],
        union=[
            "     (----]    (----]         (--------------]           ",
            " (--]   (---]          (----]    (---]    (------] (---] ",
        ],
        intersection=[
            "     (----]                   (--------------]           ",
            "        (---]                    (---]    (------]       ",
        ],
        overlap_a=[
            "                                                         ",
        ],
        overlap_b=[
            "                                                         ",
        ],
        non_overlap_a=[
            "     (----]    (----]         (--------------]           ",
        ],
        non_overlap_b=[
            " (--]   (---]          (----]    (---]    (------] (---] ",
        ],
        combine=[
            " (--](------]  (----]  (----] (------------------] (---] ",
        ],
        complement_a=[
            "          (----]    (---------]                          ",
        ],
        complement_b=[
            "    (---]   (----------]    (----]   (----]      (-]     ",
        ],
        a_diff_b=[
            "     (--]      (----]         (--]   (----]              ",
        ],
        b_diff_a=[
            " (--]     (-]          (----]                (---] (---] ",
        ],
    ),
    dict(
        A=[
            "     (----]                                              ",
            " (--]   (---]          (----]                      (---] ",
        ],
        B=[
            "               (----]         (--------------]           ",
            "                                 (---]    (------]       ",
        ],
        union=[
            "     (----]    (----]         (--------------]",
            " (--]   (---]          (----]    (---]    (------] (---] ",
        ],
        intersection=[
            "                                                         ",
        ],
        overlap_a=[
            "     (----]                                              ",
            "        (---]                                            ",
        ],
        overlap_b=[
            "                              (--------------]           ",
            "                                 (---]    (------]       ",
        ],
        non_overlap_a=[
            " (--]                  (----]                      (---] ",
        ],
        non_overlap_b=[
            "               (----]                                    ",
        ],
        combine=[
            " (--](------]  (----]  (----] (------------------] (---] ",
        ],
        complement_a=[
            "    (]      (----------]    (----------------------]     ",
        ],
        complement_b=[
            "                    (---------]                          ",
        ],
        a_diff_b=[
            " (--](------]          (----]                      (---] ",
        ],
        b_diff_a=[
            "               (----]         (------------------]       ",
        ],
    ),
]


@pytest.mark.parametrize("test_case", cases)
class TestBasicOps:
    def test_union(self, test_case):
        df_a = intervals_from_str(test_case["A"]).ivl().drop(columns=["tag"])
        df_b = intervals_from_str(test_case["B"]).ivl().drop(columns=["tag"])
        df_expected = intervals_from_str(test_case["union"]).ivl().drop(columns=["tag"])

        assert_df_interval_set_equality(df_expected, union_basic(df_a, df_b))
        assert_df_interval_set_equality(df_expected, union_basic(df_b, df_a))

    def test_intersection(self, test_case):
        df_a = intervals_from_str(test_case["A"]).ivl().drop(columns=["tag"])
        df_b = intervals_from_str(test_case["B"]).ivl().drop(columns=["tag"])
        df_expected = (
            intervals_from_str(test_case["intersection"]).ivl().drop(columns=["tag"])
        )

        assert_df_interval_set_equality(df_expected, intersection_basic(df_a, df_b))
        assert_df_interval_set_equality(df_expected, intersection_basic(df_b, df_a))

    def test_overlap(self, test_case):
        df_a = intervals_from_str(test_case["A"]).ivl().drop(columns=["tag"])
        df_b = intervals_from_str(test_case["B"]).ivl().drop(columns=["tag"])
        df_expected_a = (
            intervals_from_str(test_case["overlap_a"]).ivl().drop(columns=["tag"])
        )
        df_expected_b = (
            intervals_from_str(test_case["overlap_b"]).ivl().drop(columns=["tag"])
        )

        assert_df_interval_set_equality(df_expected_a, overlap_basic(df_a))
        assert_df_interval_set_equality(df_expected_b, overlap_basic(df_b))

    def test_non_overlap(self, test_case):
        df_a = intervals_from_str(test_case["A"]).ivl().drop(columns=["tag"])
        df_b = intervals_from_str(test_case["B"]).ivl().drop(columns=["tag"])
        df_expected_a = (
            intervals_from_str(test_case["non_overlap_a"]).ivl().drop(columns=["tag"])
        )
        df_expected_b = (
            intervals_from_str(test_case["non_overlap_b"]).ivl().drop(columns=["tag"])
        )

        assert_df_interval_set_equality(df_expected_a, non_overlap_basic(df_a))
        assert_df_interval_set_equality(df_expected_b, non_overlap_basic(df_b))

    def test_combine(self, test_case):
        df_a = intervals_from_str(test_case["A"]).ivl().drop(columns=["tag"])
        df_b = intervals_from_str(test_case["B"]).ivl().drop(columns=["tag"])
        df_expected = (
            intervals_from_str(test_case["combine"]).ivl().drop(columns=["tag"])
        )

        assert_df_interval_set_equality(
            df_expected, combine_basic(union_basic(df_a, df_b))
        )
        assert_df_interval_set_equality(
            df_expected, combine_basic(union_basic(df_b, df_a))
        )

    def test_complement(self, test_case):
        df_a = intervals_from_str(test_case["A"]).ivl().drop(columns=["tag"])
        df_b = intervals_from_str(test_case["B"]).ivl().drop(columns=["tag"])
        df_expected_a = (
            intervals_from_str(test_case["complement_a"]).ivl().drop(columns=["tag"])
        )
        df_expected_b = (
            intervals_from_str(test_case["complement_b"]).ivl().drop(columns=["tag"])
        )

        assert_df_interval_set_equality(df_expected_a, complement_basic(df_a))
        assert_df_interval_set_equality(df_expected_b, complement_basic(df_b))

    def test_diff(self, test_case):
        df_a = intervals_from_str(test_case["A"]).ivl().drop(columns=["tag"])
        df_b = intervals_from_str(test_case["B"]).ivl().drop(columns=["tag"])
        df_expected_a_diff_b = (
            intervals_from_str(test_case["a_diff_b"]).ivl().drop(columns=["tag"])
        )
        df_expected_b_diff_a = (
            intervals_from_str(test_case["b_diff_a"]).ivl().drop(columns=["tag"])
        )

        assert_df_interval_set_equality(df_expected_a_diff_b, diff_basic(df_a, df_b))
        assert_df_interval_set_equality(df_expected_b_diff_a, diff_basic(df_b, df_a))
