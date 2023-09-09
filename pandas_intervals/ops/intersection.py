from itertools import permutations

import numpy as np
import pandas as pd

from pandas_intervals.utils import apply_accessor


def _get_mask_no_ref_overlap(df_ref, df):
    ref_starts = np.sort(df_ref["start"].values)
    ref_ends = np.sort(df_ref["end"].values)
    starts = df["start"].values
    ends = df["end"].values

    # Find the index at which label starts/ends would be inserted in the
    # ref_labels starts/ends
    start_insert_idxs = np.searchsorted(ref_ends, starts)
    end_insert_idxs = np.searchsorted(ref_starts, ends)

    # When the insertion index is the same for both the interval start and end, the
    # interval has no overlapping intervals in the reference set
    mask_no_overlap_ref = start_insert_idxs == end_insert_idxs

    # Exclude when intervals are seemingly overlapping at a single point
    mask_df_start_in_ref_end = np.isin(starts, ref_ends)
    mask_df_end_in_ref_start = np.isin(ends, ref_starts)

    return mask_no_overlap_ref & ~mask_df_end_in_ref_start & ~mask_df_start_in_ref_end


@apply_accessor
def intersection(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> pd.DataFrame:
    results = []
    for df_ref, df in permutations([df_a, df_b]):
        results.append(df[~_get_mask_no_ref_overlap(df_ref, df)])

    return pd.concat(results, axis=0).drop_duplicates()


@apply_accessor
def diff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> pd.DataFrame:
    return df_a[_get_mask_no_ref_overlap(df_b, df_a)]


@apply_accessor
def symdiff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> pd.DataFrame:
    results = []
    for df_ref, df in permutations([df_a, df_b]):
        results.append(df[_get_mask_no_ref_overlap(df_ref, df)])

    return pd.concat(results, axis=0)
