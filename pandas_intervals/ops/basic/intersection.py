from itertools import product
from typing import Tuple

import pandas as pd

from pandas_intervals.utils import df_to_list

def _intervals_intersect(ivl_a: Tuple[float, float], ivl_b: Tuple[float, float]) -> bool:
    start_a, end_a, *_ = ivl_a
    start_b, end_b, *_ = ivl_b
    return (
        (start_a < start_b < end_a)
        or (start_a < end_b < end_a)
        or (start_b < start_a < end_b)
        or (start_b < end_a < end_b)
    )

def intersection(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    cols = df_a.columns
    intervals_a = df_to_list(df_a)
    intervals_b = df_to_list(df_b)

    result = set()
    for ivl_a, ivl_b in product(intervals_a, intervals_b):
        if _intervals_intersect(ivl_a, ivl_b):
            result.add(ivl_a)
            result.add(ivl_b)

    return pd.DataFrame(result, columns=cols).sort_values(["start", "end"])
            result.add(ivl_a)
            result.add(ivl_b)

    return pd.DataFrame(result, columns=cols).sort_values(["start", "end"])
