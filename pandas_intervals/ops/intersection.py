from itertools import permutations

import numpy as np
import pandas as pd


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
    return mask_no_overlap_ref


def intervals_intersection(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> pd.DataFrame:
    results = []
    for df_ref, df in permutations([df_a, df_b]):
        results.append(df[~_get_mask_no_ref_overlap(df_ref, df)])

    return pd.concat(results, axis=0)


def intervals_symdiff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> pd.DataFrame:
    results = []
    for df_ref, df in permutations([df_a, df_b]):
        results.append(df[_get_mask_no_ref_overlap(df_ref, df)])

    return pd.concat(results, axis=0)
