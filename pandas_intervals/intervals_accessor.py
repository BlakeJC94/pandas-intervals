from typing import Callable, Union, List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd


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


@pd.api.extensions.register_dataframe_accessor("intervals")
class IntervalsAccessor(FieldsTrait):
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

    def __init__(self, pandas_obj=None):
        if pandas_obj.empty or pandas_obj.columns.empty:
            pandas_obj = self.empty()
        else:
            pandas_obj = pandas_obj.rename(
                columns={i: col for i, col in enumerate(self.cols)}
            )
            self._validate(pandas_obj)
            pandas_obj = self._fill_defaults_for_cols(pandas_obj)
            pandas_obj = self._set_types(pandas_obj)
            pandas_obj = self._sort_columns(pandas_obj)
        self._obj = pandas_obj

    def __call__(self):
        return self._obj

    @classmethod
    def empty(cls) -> pd.DataFrame:
        dtype = [(name, kind) for name, kind, _ in cls.fields]
        return pd.DataFrame(np.empty(0, dtype=np.dtype(dtype)))

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

    @property
    def durations(self) -> pd.Series:
        starts = self._obj["start"]
        ends = self._obj["end"]
        return ends - starts

    def plot(self):
        # plot durations using plotly
        pass

    def union(self, other):  # TODO Make this implementation accessor-name independent
        return intervals_union(
            self._obj, other.intervals(), sort_cols=self.additional_cols
        )

    # TODO intersection
    # TODO intersection


def intervals_union(
    vf_a: pd.DataFrame, vf_b: pd.DataFrame, sort_cols: Optional[List[str]]
) -> IntervalsFrame:
    sort_cols = sort_cols if sort_cols is not None else []
    return (
        pd.concat([vf_a, vf_b], axis=0)
        .sort_values(["start", *sort_cols])
        .drop_duplicates()
    )
