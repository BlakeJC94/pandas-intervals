import inspect
from typing import Callable, Iterable, List, Tuple, Any, Optional

import pandas as pd


def sort_intervals(
    df: pd.DataFrame,
    sort_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    sort_cols = sort_cols if sort_cols is not None else []
    result = df.sort_values(["start", "end", *sort_cols])
    return result


def comma_join(x: List[str]) -> str:
    return ", ".join(sorted({i.strip() for n in x if n for i in n.split(",")}))


def _apply_operation_to_groups(
    operation: Callable,
    dfs: List[pd.DataFrame],
    groupby_cols: List[str],
    aggregations=None,
    **kwargs,
) -> pd.DataFrame:
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
    return pd.concat(results, axis=0)


def _df_groups(
    dfs: List[pd.DataFrame],
    groupby_cols: Optional[List[str]] = None,
) -> Iterable[Tuple[Any, List[pd.DataFrame]]]:
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
