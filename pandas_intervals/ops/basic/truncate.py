import pandas as pd

from .combine import combine
from pandas_intervals.utils import apply_accessor, df_to_list


@apply_accessor
def truncate(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
):
    if len(df_a) == 0 or len(df_b) == 0:
        return df_a

    df_a = df_a.sort_values(["start", "end"])
    df_b = combine(df_b[["start", "end"]]).sort_values("start")

    # Remove zero duration df_b that are equal to start/end of df_a (no effect)
    _mask = ((df_b["end"] - df_b["start"]) == 0) & (
        df_b["start"].isin(df_a["start"]) | df_b["start"].isin(df_a["end"])
    )
    df_b = df_b[~_mask]

    # Track the index as a column
    df_a = df_a.assign(_index=df_a.index)

    intervals_a = df_to_list(df_a)
    intervals_b = df_to_list(df_b)

    results = []
    for start_a, end_a, *metadata in intervals_a:
        keep_label = True

        # check overlap of selected interval in `a` with bounding intervals in B
        for start_b, end_b, *_ in intervals_b:
            # A :          (----]
            # B : (----]
            # If `b` is strictly before `a`, check the interval in `B`
            if end_b < start_a:
                continue

            # A :          (----]
            # B :                   (----]
            # If `b` is strictly after the selected label, go to next interval in `A`
            if end_a < start_b:
                break

            # A :          (----]
            # B :       (----------]
            # If `b` contains `a`, discard `a` and go to next interval in `A`
            if start_b < start_a and end_a <= end_b:
                keep_label = False
                break

            # A :          (------]
            # B :      (------]
            # If `b` overlaps start of `a`, clip `a` start and check next interval in `B`
            if start_b < start_a and end_b <= end_a:
                start_a = end_b
                continue

            # A :         (-----------...
            # B :            (----]
            # If `b` is contained in `a`, create interval, clip `a`, and check next interval in `b`
            if start_a <= start_b and end_b < end_a:
                results.append((start_a, start_b, *metadata))
                start_a = end_b
                continue

            # A :            (------]
            # B :                (------]
            # If `b` overlaps end of `a`, clip end of `a` and go to next interval in `A`
            if start_a <= start_b and end_a <= end_b:
                end_a = start_b
                break

        if keep_label:
            results.append((start_a, end_a, *metadata))

    return (
        pd.DataFrame(results, columns=df_a.columns)
        .set_index("_index")
        .rename_axis(index=None)
    )
