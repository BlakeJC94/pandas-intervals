from functools import partial
from typing import Callable, Union, List, Dict, Optional

import numpy as np
import pandas as pd

try:
    import plotly
except ImportError:
    plotly = None

from pandas_intervals.ops import (
    basic,
    overlap,
    non_overlap,
    intersection,
    symdiff,
    complement,
    combine,
    truncate,
    nearest,
)
from .vis import plot_intervals
from .utils import _apply_operation_to_groups, sort_intervals


class FieldsTrait:
    """Mixin used to set the essential columns/types of a DataFrame of interval-like objects.

    Provides a scaffold for extending this further to add more columns with custom names, default
    values, and aggregation methods.
    """

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

    @classmethod
    @property
    def apply_to_groups(cls) -> Callable:
        return partial(
            _apply_operation_to_groups,
            groupby_cols=cls.groupby_cols,
            aggregations=cls.aggregations,
        )


class FormatTrait:
    """Mixin used for providing a `format` method for an Accessor class.

    When the accessor is used, the `__init__` method will call the `format` method defined to
    * Check DataFrame has required columns,
    * Add additional cols with configured defaults,
    * Check types and change if needed.

    This will ensure that all the operations on DataFrames of interval-like objects have controlled
    types and column names and orders across all operations used throughout this extension.
    """

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
        if any(c not in cls.additional_cols for c in cls.default_values):
            raise ValueError(
                f"{cls.__name__} contains keys in `default_values` not located in `additional_cols`."
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

    def __init__(self, pandas_obj: Optional[pd.DataFrame] = None):
        self.df = self.format(pandas_obj)
        self._basic = False

    def __call__(self):
        return self.df

    @property
    def durations(self) -> pd.Series:
        starts = self.df["start"]
        ends = self.df["end"]
        return ends - starts

    @property
    def span(self) -> float:
        return self.df["end"].max() - self.df["start"].min()

    @property
    def basic(self):
        self._basic = True
        return self

    def plot(  # IDEA: matplotlib impl?
        self,
        groupby_cols: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        **layout_kwargs,
    ):
        if plotly is None:
            raise ImportError("Plotting intervals requires `plotly` to be installed")

        groupby_cols = groupby_cols or self.groupby_cols
        return plot_intervals(self.df, groupby_cols, colors, **layout_kwargs)

    def sort(self) -> pd.DataFrame:
        results = sort_intervals(
            self.df,
            sort_cols=self.additional_cols,
        )
        return results.reset_index(drop=True)

    # TODO Test
    def window(
        self,
        left_bound: Optional[float] = None,
        right_bound: Optional[float] = None,
        strict: bool = False,
    ) -> pd.DataFrame:
        if left_bound is None:
            left_bound = self.df["start"].min()
        if right_bound is None:
            left_bound = self.df["end"].max()

        if not strict:
            mask = (self.df["end"] >= left_bound) & (self.df["start"] <= right_bound)
        else:
            mask = (self.df["start"] >= left_bound) & (self.df["end"] <= right_bound)

        return self.df.loc[mask]

    # TODO Test
    def contains(self, df: pd.DataFrame) -> bool:
        df = df.drop_duplicates()
        df_all = self.union(df)
        return len(self.df.drop_duplicates()) == len(df_all)

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

        self.df["start"] = self.df["start"] - (left_pad or 0)
        self.df["end"] = self.df["end"] + (right_pad or 0)
        return self.df.loc[self.df["end"] - self.df["start"] >= 0]

    def overlap(self) -> pd.DataFrame:
        operation = basic.overlap if self._basic else overlap
        return self.apply_to_groups(
            operation,
            [self.df],
        )

    def non_overlap(self) -> pd.DataFrame:
        operation = basic.non_overlap if self._basic else non_overlap
        return self.apply_to_groups(
            operation,
            [self.df],
        )

    def complement(
        self,
        left_bound: Optional[float] = None,
        right_bound: Optional[float] = None,
    ):
        operation = basic.complement if self._basic else complement
        return self.apply_to_groups(
            operation,
            [self.df],
            left_bound=left_bound,
            right_bound=right_bound,
        )

    def union(self, *dfs) -> pd.DataFrame:
        interval_sets = [self.df, *[self.format(df) for df in dfs]]
        return pd.concat(interval_sets, axis=0).drop_duplicates()

    def intersection(self, df: pd.DataFrame) -> pd.DataFrame:
        operation = basic.intersection if self._basic else intersection
        return self.apply_to_groups(
            operation,
            [self.df, self.format(df)],
        )

    def symdiff(self, df: pd.DataFrame) -> pd.DataFrame:
        operation = basic.symdiff if self._basic else symdiff
        return self.apply_to_groups(
            operation,
            [self.df, self.format(df)],
        )

    def combine(self, *dfs) -> pd.DataFrame:
        operation = basic.combine if self._basic else combine
        return self.apply_to_groups(
            operation,
            [self.union(*dfs)],
        )

    def truncate(self, df: pd.DataFrame):
        operation = basic.truncate if self._basic else truncate
        return self.apply_to_groups(
            operation,
            [self.df, self.format(df)],
        )

    def nearest(self, df: pd.DataFrame):
        operation = basic.nearest if self._basic else nearest
        return self.apply_to_groups(
            operation,
            [self.df, self.format(df)],
        )
