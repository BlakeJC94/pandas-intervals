from typing import Union, Optional, Dict, Callable

import pandas as pd

from pandas_intervals.utils import apply_accessor


@apply_accessor
def combine(
    df: pd.DataFrame,
    aggregations: Optional[Dict[str, Union[str, Callable]]] = None,
):
    if df.empty:
        return df

    aggregations = aggregations or {"start": "min", "end": "max"}
    aggregations.update({c: "first" for c in df.columns if c not in aggregations})
    df_sorted = df.sort_values("start")

    # Loop over labels and compare each to the previous label to find labels to combine
    group_inds = []
    ind, interval_end_time = 0, 0
    for start, end in df_sorted[["start", "end"]].values:
        # If interval is within previous label, combine them
        if start < interval_end_time:
            interval_end_time = max(interval_end_time, end)
            group_inds.append(ind)
        # If not, start a new interval
        else:
            interval_end_time = end
            ind += 1
            group_inds.append(ind)

    return df_sorted.groupby(group_inds).agg(aggregations)
