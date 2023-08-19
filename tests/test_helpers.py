import pytest

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
            a_union_b=[
                "     (----]    (----]         (--------------]           ",
                " (--]   (---]          (----]    (---]    (------] (---] ",
            ],
        ),
        dict(
            a=[
                "     (----]                                              ",
                " (--]   (---]          (----]                      (---] ",
            ],
            b=[
                "               (----]         (--------------]           ",
                "                                 (---]    (------]       ",
            ],
            a_union_b=[
                "     (----]    (----]         (--------------]",
                " (--]   (---]          (----]    (---]    (------] (---] ",
            ],
        ),
    ],
)
def test_union(test_case):
    df_a = intervals_from_str(test_case["a"])
    df_b = intervals_from_str(test_case["b"])
    df_expected = intervals_from_str(test_case["a_union_b"])
    assert_df_interval_set_equality(df_expected, union_basic(df_a, df_b))
    assert_df_interval_set_equality(df_expected, union_basic(df_b, df_a))


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
def test_intersection(test_case):
    df_a = intervals_from_str(test_case["a"])
    df_b = intervals_from_str(test_case["b"])
    df_expected = intervals_from_str(test_case["a_intersection_b"])
    assert_df_interval_set_equality(df_expected, intersection_basic(df_a, df_b))
    assert_df_interval_set_equality(df_expected, intersection_basic(df_b, df_a))


@pytest.mark.parametrize(
    "test_case",
    [
        dict(
            a=[
                "     (----]    (----]         (--------------]           ",
            ],
            overlap_a=[
                "                                                         ",
            ],
            non_overlap_a=[
                "     (----]    (----]         (--------------]           ",
            ],
        ),
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
def test_overlap(test_case):
    df_a = intervals_from_str(test_case["a"])
    df_expected_overlap_a = intervals_from_str(test_case["overlap_a"])
    df_expected_non_overlap_a = intervals_from_str(test_case["non_overlap_a"])
    assert_df_interval_set_equality(df_expected_overlap_a, overlap_basic(df_a))
    assert_df_interval_set_equality(df_expected_non_overlap_a, non_overlap_basic(df_a))


@pytest.mark.parametrize(
    "test_case",
    [
        dict(
            a=[
                "     (----]    (----]         (--------------]           ",
            ],
            combine_a=[
                "     (----]    (----]         (--------------]           ",
            ],
        ),
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
def test_combine(test_case):
    df_a = intervals_from_str(test_case["a"])
    df_expected_combine_a = intervals_from_str(test_case["combine_a"])
    assert_df_interval_set_equality(df_expected_combine_a, combine_basic(df_a))


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
def test_complement(test_case):
    df_a = intervals_from_str(test_case["a"])
    df_expected_a = intervals_from_str(test_case["complement_a"])
    assert_df_interval_set_equality(df_expected_a, complement_basic(df_a))


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
            b_diff_a=[
                " (--]     (-]          (----]                (---] (---] ",
            ],
        ),
        dict(
            a=[
                "     (----]                                              ",
                " (--]   (---]          (----]                      (---] ",
            ],
            b=[
                "               (----]         (--------------]           ",
                "                                 (---]    (------]       ",
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
    ],
)
def test_diff(test_case):
    df_a = intervals_from_str(test_case["a"])
    df_b = intervals_from_str(test_case["b"])
    df_expected_a_diff_b = intervals_from_str(test_case["a_diff_b"])
    df_expected_b_diff_a = intervals_from_str(test_case["b_diff_a"])
    assert_df_interval_set_equality(df_expected_a_diff_b, diff_basic(df_a, df_b))
    assert_df_interval_set_equality(df_expected_b_diff_a, diff_basic(df_b, df_a))
