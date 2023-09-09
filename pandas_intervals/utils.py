import inspect
from typing import Callable, Iterable, List, Tuple, Any, Optional, Set

import pandas as pd


def sort_intervals(
    df: pd.DataFrame,
    sort_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    sort_cols = sort_cols or []
    result = df.sort_values(["start", "end", *sort_cols])
    return result


def df_to_set(df: pd.DataFrame) -> Set[List[Any]]:
    return set(df.itertuples(index=False, name=None))


def df_to_list(df: pd.DataFrame) -> List[Tuple[Any]]:
    return list(df.itertuples(index=False, name=None))


# TODO Turn this into a decorator which optially does this when supplied with accessor
def _apply_operation_to_groups(
    operation: Callable,
    dfs: List[pd.DataFrame],
    additional_cols: Optional[List[str]] = None,
    groupby_cols: Optional[List[str]] = None,
    aggregations: Optional[Dict[str, Agg]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Helper function to wrap an operation, apply it to `groupby` results acuss multiple
    DataFrames, apply the `aggregations` kwarg if present in operation argspec, concatentate and
    return results.
    """
    groupby_cols = groupby_cols or []

    results = []
    for _, df_groups in _df_groups(dfs, groupby_cols=groupby_cols):
        if (
            aggregations is not None
            and "aggregations" in inspect.getfullargspec(operation).args
        ):
            result = operation(*df_groups, aggregations=aggregations, **kwargs)
        else:
            result = operation(*df_groups, **kwargs)
        results.append(result)

    if any(not isinstance(r, (pd.DataFrame, pd.Series)) for r in results):
        raise ValueError("Expected to get a list of `pd.DataFrame` or `pd.Series`.")

    results = pd.concat(results, axis=0)
    if isinstance(results, pd.DataFrame):
        results = sort_intervals(results, sort_cols=additional_cols)
    return results


def _df_groups(
    dfs: List[pd.DataFrame],
    groupby_cols: Optional[List[str]] = None,
) -> Iterable[Tuple[Any, List[pd.DataFrame]]]:
    """Iterator for applying a `groupby` operation to multiple DataFrames."""
    groupby_cols = groupby_cols or []
    if len(groupby_cols) == 0:
        yield (None, dfs)
    else:
        n_dfs = len(dfs)
        for i in range(n_dfs):
            dfs[i]["_arg"] = i

        df = pd.concat(dfs)
        if len(groupby_cols) == 1:
            groupby_cols = groupby_cols[0]
        for group, df_group in df.groupby(groupby_cols):
            result = [
                df_group[(df_group["_arg"] == i)].drop(columns=["_arg"])
                for i in range(n_dfs)
            ]
            yield (group, result)
