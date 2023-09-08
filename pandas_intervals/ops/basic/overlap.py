
import numpy as np
import pandas as pd

from pandas_intervals.utils import df_to_list


def _get_overlap_mask(df_a: pd.DataFrame) -> np.ndarray:
    intervals_a = df_to_list(df_a)

    mask = []
    for start_a, end_a, *_ in intervals_a:
        overlap = False
        for start_b, end_b, *_ in intervals_a:
            if end_b < start_a:
                continue
            if end_a < start_b:
                break
            if (
                (start_a < start_b < end_a)
                or (start_a < end_b < end_a)
                or (start_b < start_a < end_b)
                or (start_b < end_a < end_b)
            ):
                overlap = True
                break
        mask.append(overlap)

    return np.array(mask)


def overlap(df_a: pd.DataFrame) -> pd.DataFrame:
    if df_a.empty:
        return df_a
    return df_a.loc[_get_overlap_mask(df_a)]


def non_overlap(df_a: pd.DataFrame) -> pd.DataFrame:
    if df_a.empty:
        return df_a
    return df_a.loc[~_get_overlap_mask(df_a)]
