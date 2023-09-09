from typing import Callable, Union, Dict, Optional

import pandas as pd

from .overlap import _get_overlapping_mask
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

    # Filter out non_overlapping intervals to those that overlap
    overlap_mask = _get_overlapping_mask(df_sorted)
    df_sorted_non_overlap = df_sorted.loc[overlap_mask == 0]
    df_sorted_overlap = df_sorted.loc[overlap_mask > 0]

    # Aggregate overlaps
    group_inds = overlap_mask[overlap_mask > 0]
    df_sorted_overlap_agg = df_sorted_overlap.groupby(group_inds).agg(aggregations)

    return pd.concat([df_sorted_non_overlap, df_sorted_overlap_agg], axis=0)
