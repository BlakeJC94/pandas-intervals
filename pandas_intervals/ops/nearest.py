import logging

import numpy as np
import pandas as pd

from .intersection import _get_mask_no_ref_overlap
from pandas_intervals.utils import apply_accessor


logger = logging.getLogger(__name__)


@apply_accessor
def nearest(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.Series:
    """Given DataFrames A and B containing columns `["start", "end"]` where each row represents an
    interval, calculate the distance to the closest interval in B for each row in A.

    Args:
        df_a: DataFrame representing intervals. Must contain columns `["start", "end"]`.
        df_b: DataFrame representing intervals. Must contain columns `["start", "end"]`.

    Returns:
        DataFrame with column `"min_dist"` and index equal to index of A.
    """
    if len(df_a) == 0 or len(df_b) == 0:
        logger.warning(f"Recieved empty intervals DataFrame.")
        return pd.Series(
            np.full((len(df_a)), np.inf),
            index=df_a.index,
        )

    (starts_a, ends_a) = df_a[["start", "end"]].to_numpy().T
    (starts_b, ends_b) = df_b[["start", "end"]].to_numpy().T

    # For each end_a, find the previous start_b
    idxs_b_left_of_a = np.searchsorted(np.sort(ends_b), starts_a) - 1
    mask_idxs_b_left_inf = idxs_b_left_of_a == -1
    # Handle cases when there is no interval in B left of intervals of A
    left_dist = np.abs(ends_b[idxs_b_left_of_a] - starts_a)
    left_dist[mask_idxs_b_left_inf] = np.inf

    # For each start_a, find the next end_b (set idx_b to -1 if nothing right of a)
    idxs_b_right_of_a = np.searchsorted(np.sort(starts_b), ends_a)
    mask_idxs_b_right_inf = idxs_b_right_of_a == len(starts_b)
    idxs_b_right_of_a[mask_idxs_b_right_inf] = -1
    # Handle cases when there is no interval in B right of intervals of A
    right_dist = np.abs(starts_b[idxs_b_right_of_a] - ends_a)
    right_dist[mask_idxs_b_right_inf] = np.inf

    # Map negative distances and overlaps to 0
    min_dist = np.minimum(left_dist, right_dist).clip(0)
    mask_non_overlap = _get_mask_no_ref_overlap(df_b, df_a)
    min_dist[~mask_non_overlap] = 0

    return pd.Series(
        min_dist,
        index=df_a.index,
    )
