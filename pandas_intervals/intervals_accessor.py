from typing import Callable, Iterable, Union, List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd

from .interval_ops import (
    intervals_union,
    intervals_intersection,
    intervals_complement,
    intervals_overlap,
    intervals_non_overlap,
    intervals_combine,
    intervals_difference,
)


# Every time df.intervals is called, `init` is called!
# Potentialy this is really good for ensuring that the types are always valid

# When I use the accessor,
# * Check dataframe has required columns
# * Add additional cols w/defaults
# * Check types and change if needed


class FieldsTrait:
    _required_fields = [("start", "float64", "min"), ("end", "float64", "max")]
    additional_fields = []
    default_values = {}

    @classmethod
    @property
    def fields(cls) -> Dict[str, Union[str, Callable]]:
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
    def aggregations(cls) -> Dict[str, Union[str, Callable]]:
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
    @classmethod
    def format(cls, pandas_obj: pd.DataFrame) -> pd.DataFrame:
        if pandas_obj.columns.empty:
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


@pd.api.extensions.register_dataframe_accessor("ivl")
class IntervalsAccessor(FieldsTrait, FormatTrait):
    """A DataFrame accessor for frames containing intervals (columns "start" and "end").

    Invoking this accessor on a DataFrame will check the frame is a valid representation of
    intervals, set types on the columns if needed, and sort the columns in a standard order so that
    column 0 and 1 are "start" and "end".

    To simply validate and format a DataFrame containing intervals, call this accessor on a
    DataFrame:

        >>> df.regions()

    Can be extended by subclassing `IntervalsAccessor` and adding a decorator. For example, if we
    want to create an intervals accessor called `"regions"` which
        * Has 2 extra columns ("tag" and "note"),
        * Column "tag" must be specified, but "note" is optional,
        * Column "tag" is an integer, and "note" is a string,
        * Aggregations are done across different values of "tag", and "note" values are combined
            into a comma-separated string.

    We can accomplish this in a relatively small class:

        >>> @pd.api.extensions.register_dataframe_accessor("regions")  # Name of new accessor
        ... class RegionsAccessor(IntervalsAccessor):
        ...
        ...     # Additional required columns can be specified in a list of tuple
        ...     # where each tuple is `(column_name, dtype, aggregation)`
        ...     additional_cols = [
        ...         ("tag", "int64", "groupby"),
        ...         ("note", "object", lambda x: ','.join(x)),
        ...     ]
        ...
        ...     # Default values for columns can be specified as a dictionary,
        ...     # columns that don't appear in this list are assumed to be necessary
        ...     default_values = {
        ...          "note": "",
        ...     }

    """

    def __init__(self, pandas_obj: pd.DataFrame):
        self.df = self.format(pandas_obj)

    def __call__(self):
        return self.df

    @property
    def durations(self) -> pd.Series:
        starts = self.df["start"]
        ends = self.df["end"]
        return ends - starts

    # TODO plot durations using plotly
    # TODO format and plot other dfs on rows
    def plot(self, *dfs):
        # TODO raise if plotly not installed
        pass

    def sort(self) -> pd.DataFrame:
        results = sort_intervals(
            self.df,
            sort_cols=self.additional_cols,
        )
        return results.reset_index(drop=True)

    def pad(
        self,
        pad: Optional[float] = None,
        left_pad: Optional[float] = None,
        right_pad: Optional[float] = None,
    ):
        if pad is not None:
            if left_pad is not None or right_pad is not None:
                raise ValueError("Either use `pad`, or `left_pad`/`right_pad`.")
            left_pad, right_pad = pad, pad

        self.df["start"] = self.df["start"] - left_pad
        self.df["end"] = self.df["end"] + right_pad
        return self.df

    def unpad(
        self,
        unpad: Optional[float] = None,
        left_unpad: Optional[float] = None,
        right_unpad: Optional[float] = None,
    ):
        if unpad is not None:
            if left_unpad is not None or right_unpad is not None:
                raise ValueError("Either use `unpad`, or `left_unpad`/`right_unpad`.")
            left_unpad, right_unpad = unpad, unpad

        self.df["start"] = self.df["start"] + left_unpad
        self.df["end"] = self.df["end"] - right_unpad
        return self.df.loc[self.df["end"] - self.df["start"] >= 0]

    def overlap(self) -> pd.DataFrame:
        return self._apply_operation_to_groups(
            intervals_overlap,
            [self.df],
        )

    def non_overlap(self) -> pd.DataFrame:
        return self._apply_operation_to_groups(
            intervals_non_overlap,
            [self.df],
        )

    def complement(
        self,
        left_bound: Optional[float] = None,
        right_bound: Optional[float] = None,
    ):
        return self._apply_operation_to_groups(
            intervals_complement,
            [self.df],
            left_bound=left_bound,
            right_bound=right_bound,
        )

    def union(self, *dfs) -> pd.DataFrame:
        interval_sets = [self.df, *[self.format(df) for df in dfs]]
        return intervals_union(interval_sets)

    def intersection(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._apply_operation_to_groups(
            intervals_intersection,
            [self.df, self.format(df)],
        )

    def combine(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._apply_operation_to_groups(
            intervals_combine,
            [self.union(self.df, df)],
            aggregations=self.aggregations,
        )

    def diff(self, df: pd.DataFrame):
        return self._apply_operation_to_groups(
            intervals_difference,
            [self.df, self.format(df)],
        )

    def _apply_operation_to_groups(
        self,
        operation: Callable,
        dfs: List[pd.DataFrame],
        **kwargs,
    ) -> pd.DataFrame:
        results = []
        for _, df_groups in _df_groups(dfs, groupby_cols=self.groupby_cols):
            results.append(operation(*df_groups, **kwargs))
        return pd.concat(results, axis=0)


def sort_intervals(
    df: pd.DataFrame,
    sort_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    sort_cols = sort_cols if sort_cols is not None else []
    result = df.sort_values(["start", "end", *sort_cols])
    return result




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
        for group, df_group in df.groupby(groupby_cols):
            result = [
                df_group[(df_group["_arg"] == i)].drop(columns=["_arg"])
                for i in range(n_dfs)
            ]
            yield (group, result)
