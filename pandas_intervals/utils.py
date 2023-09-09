import inspect
from functools import partial
from typing import Callable, Iterable, List, Tuple, Any, Optional, Set, Union, Dict

import pandas as pd
import numpy as np

Agg = Union[str, Callable[[List[Any]], Any]]
Field = Tuple[str, str, Agg]


class FieldsTrait:
    """Mixin used to set the essential columns/types of a DataFrame of interval-like objects.

    Provides a scaffold for extending this further to add more columns with custom names, default
    values, and aggregation methods.
    """

    _required_fields: List[Field] = [
        ("start", "float64", "min"),
        ("end", "float64", "max"),
    ]
    additional_fields: List[Field] = []
    default_values: Dict[str, Any] = {}

    @classmethod
    @property
    def fields(cls) -> List[Field]:
        return [*cls._required_fields, *cls.additional_fields]

    @classmethod
    @property
    def cols(cls) -> List[str]:
        return [i[0] for i in cls.fields]

    @classmethod
    @property
    def required_cols(cls) -> List[str]:
        return [i[0] for i in cls.fields if i[0] not in cls.default_values]

    @classmethod
    @property
    def additional_cols(cls) -> List[str]:
        return [i[0] for i in cls.fields if i[0] in cls.default_values]

    @classmethod
    @property
    def aggregations(cls) -> Dict[str, Agg]:
        return {col: agg for col, _, agg in cls.fields if agg != "groupby"}

    @classmethod
    @property
    def groupby_cols(cls) -> List[str]:
        return [col for col, _, agg in cls.fields if agg == "groupby"]

    @classmethod
    def empty(cls) -> pd.DataFrame:
        dtype = [(name, kind) for name, kind, _ in cls.fields]
        return pd.DataFrame(np.empty(0, dtype=np.dtype(dtype)))


class FormatTrait:
    """Mixin used for providing a `format` method for an Accessor class."""

    @classmethod
    def format(cls, pandas_obj: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if pandas_obj is None or pandas_obj.columns.empty:
            return cls.empty()
        pandas_obj = pandas_obj.rename(
            columns={i: col for i, col in enumerate(cls.cols)}
        )
        cls._validate(pandas_obj)
        pandas_obj = cls._fill_defaults_for_cols(pandas_obj)
        pandas_obj = cls._set_types(pandas_obj)
        pandas_obj = cls._sort_columns(pandas_obj)
        return pandas_obj

    @classmethod
    def _validate(cls, obj):
        invalid_additional_cols = [
            c for c in cls.additional_cols if c in cls.required_cols
        ]
        if invalid_additional_cols:
            breakpoint()
            raise AttributeError(
                f"{cls.__name__} contains keys in `additional_fields` that clash with required "
                f"columns: {invalid_additional_cols}. Please remove or rename these columns in "
                f"the `additional_fields` class attribute."
            )

        missing_cols_default_vals = [
            c for c in cls.default_values if c not in cls.additional_cols
        ]
        if missing_cols_default_vals:
            raise AttributeError(
                f"{cls.__name__} contains keys in `default_values` not located in "
                f"`additional_cols`: {missing_cols_default_vals}. Please add these keys "
                f"with default values to the `default_values` class attribute."
            )

        missing_cols = [col for col in cls.required_cols if col not in obj]
        if len(missing_cols) > 0:
            raise ValueError(
                f"DataFrame missing required column(s) '{', '.join(missing_cols)}'."
            )

        if (obj["end"] - obj["start"] < 0).any():
            raise ValueError("DataFrame contains invalid intervals.")

    @classmethod
    def _fill_defaults_for_cols(cls, obj: pd.DataFrame) -> pd.DataFrame:
        for (
            col,
            default_val,
        ) in cls.default_values.items():
            obj[col] = obj[col].fillna(default_val) if col in obj else default_val
        return obj

    @classmethod
    def _set_types(cls, obj: pd.DataFrame) -> pd.DataFrame:
        for col, kind, _ in cls.fields:
            if obj.dtypes[col] != kind:
                obj[col] = obj[col].astype(kind)
        return obj

    @classmethod
    def _sort_columns(cls, obj: pd.DataFrame) -> pd.DataFrame:
        extra_cols = [col for col in obj.columns if col not in cls.cols]
        return obj[[*cls.cols, *extra_cols]]


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


def apply_accessor(func: Callable):
    """Decorator for applying an intervals operation across groups automatically when given an
    accessor.
    """
    def inner(*dfs, accessor=None, **kwargs):
        if accessor is None:
            return func(*dfs, **kwargs)

        groupby_cols = accessor.groupby_cols
        aggregations = accessor.aggregations
        additional_cols = accessor.additional_cols

        results = []
        for _, df_groups in _df_groups(dfs, groupby_cols=groupby_cols):
            if "aggregations" in inspect.getfullargspec(func).args:
                result = func(*df_groups, aggregations=aggregations, **kwargs)
            else:
                result = func(*df_groups, **kwargs)
            results.append(result)

        if any(not isinstance(r, (pd.DataFrame, pd.Series)) for r in results):
            raise ValueError("Expected to get a list of `pd.DataFrame` or `pd.Series`.")

        results = pd.concat(results, axis=0)
        if isinstance(results, pd.DataFrame):
            results = sort_intervals(results, sort_cols=additional_cols)

        return results

    return inner


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
