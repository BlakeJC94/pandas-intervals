from itertools import product

import pandas as pd

from pandas_intervals.utils import df_to_list


def intervals_intersection(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    cols = df_a.columns
    intervals_a = df_to_list(df_a)
    intervals_b = df_to_list(df_b)

    result = set()
    for ivl_a, ivl_b in product(intervals_a, intervals_b):
        start_a, end_a, *_ = ivl_a
        start_b, end_b, *_ = ivl_b
        if (
            (start_a < start_b < end_a)
            or (start_a < end_b < end_a)
            or (start_b < start_a < end_b)
            or (start_b < end_a < end_b)
        ):
            result.add(ivl_a)
            result.add(ivl_b)

    return pd.DataFrame(result, columns=cols).sort_values(["start", "end"])
