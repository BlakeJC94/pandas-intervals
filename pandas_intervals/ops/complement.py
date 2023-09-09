from typing import Callable, Union, List, Dict, Optional

import pandas as pd

from .combine import combine
from pandas_intervals.utils import apply_accessor


@apply_accessor
def complement(
    df: pd.DataFrame,
    aggregations: Optional[Dict[str, Union[str, Callable]]] = None,
    left_bound: Optional[float] = None,
    right_bound: Optional[float] = None,
):
    if len(df) == 0:
        return df

    aggregations = aggregations or {"start": "min", "end": "max"}
    aggregations.update({c: "first" for c in df.columns if c not in aggregations})
    df = combine(df, aggregations=aggregations).sort_values("start")

    (start_first, end_first), metadata_first = df.iloc[0, :2], df.iloc[0, 2:]
    (_start_last, end_last), metadata_last = df.iloc[-1, :2], df.iloc[-1, 2:]
    if left_bound is None:
        left_bound = end_first
    if right_bound is None:
        right_bound = end_last

    edges: List[pd.Series] = []
    if left_bound < start_first:
        left_edge = metadata_first
        left_edge["start"] = left_bound
        left_edge["end"] = start_first
        left_edge = left_edge[df.columns]
        edges.append(left_edge)
    if end_last < right_bound:
        right_edge = metadata_last
        right_edge["start"] = end_last
        right_edge["end"] = right_bound
        right_edge = right_edge[df.columns]
        edges.append(right_edge)

    df_c = df.rolling(2).agg(aggregations).iloc[1:]
    df_c["start"] = df.iloc[:-1, 1].to_numpy()
    df_c["end"] = df.iloc[1:, 0].to_numpy()
    df_c = df_c[df.columns]

    return pd.concat([pd.DataFrame(edges), df_c], axis=0)
