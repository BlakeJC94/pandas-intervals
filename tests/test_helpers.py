import pytest

from tests.helpers import (
    assert_df_interval_set_equality,
    intervals_from_str,
    union_basic,
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
