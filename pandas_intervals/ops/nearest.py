import numpy as np
import pandas as pd


def intervals_nearest(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Given DataFrames A and B containing columns `["start", "end"]` where each row represents an
    interval, calculate the distance to the closest interval in B for each row in A.

    Args:
        df_a: DataFrame representing intervals. Must contain columns `["start", "end]`.
        df_b: DataFrame representing intervals. Must contain columns `["start", "end]`.

    Returns:
        DataFrame A with an additional column `"min_dist"`.
    """
    (starts_a, ends_a) = df_a[["start", "end"]].to_numpy().T
    (starts_b, ends_b) = df_b[["start", "end"]].to_numpy().T

    # For each end_a, find the previous start_b
    idxs_b_left_of_a = np.searchsorted(np.sort(ends_b), starts_a) - 1
    mask_idxs_b_left_inf = idxs_b_left_of_a == -1
    idxs_b_left_of_a[mask_idxs_b_left_inf] = -1
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

    # Map negative distances to 0
    min_dist = np.minimum(left_dist, right_dist).clip(0)
    return df_a.assign(min_dist=min_dist)
