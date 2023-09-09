import pandas as pd

from pandas_intervals.utils import apply_accessor, df_to_list


@apply_accessor
def nearest(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> pd.Series:
    intervals_a = df_to_list(df_a)
    intervals_b = df_to_list(df_b)

    results = []
    for start_a, end_a, *_ in intervals_a:
        min_dist = float("inf")
        for start_b, end_b, *_ in intervals_b:
            if end_b <= start_a:
                min_dist = min(min_dist, start_a - end_b)
            if end_a <= start_b:
                min_dist = min(min_dist, start_b - end_a)
            # Intersections should result in zero distances
            if (
                (start_a < start_b < end_a)
                or (start_a < end_b < end_a)
                or (start_b < start_a < end_b)
                or (start_b < end_a < end_b)
            ):
                min_dist = 0
        results.append(min_dist)

    return pd.Series(
        results,
        index=df_a.index,
    )
