from typing import Union, Optional, Dict, Callable

import pandas as pd

from .combine import combine
from pandas_intervals.utils import df_to_list, apply_accessor


@apply_accessor
def complement(
    df_a: pd.DataFrame,
    aggregations: Optional[Dict[str, Union[str, Callable]]] = None,
    left_bound: Optional[float] = None,
    right_bound: Optional[float] = None,
):
    if len(df_a) == 0:
        return df_a

    df_a = combine(df_a, aggregations=aggregations).sort_values("start")
    intervals = df_to_list(df_a)

    start_first, end_first, *metadata_first = intervals[0]
    _start_last, end_last, *metadata_last = intervals[-1]
    if left_bound is None:
        left_bound = end_first
    if right_bound is None:
        right_bound = end_last

    results = []
    if left_bound < start_first:
        results.append((left_bound, start_first, *metadata_first))

    for i in range(len(intervals) - 1):
        _start_prev, end_prev, *metadata = intervals[i]
        start_next, _end_next, *_ = intervals[i + 1]
        results.append((end_prev, start_next, *metadata))

    if end_last < right_bound:
        results.append((end_last, right_bound, *metadata_last))

    return pd.DataFrame(results, columns=df_a.columns)
