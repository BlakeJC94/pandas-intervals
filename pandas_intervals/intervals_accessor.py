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


@pd.api.extensions.register_dataframe_accessor("ivl")
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

    # TODO Consider if formatting should be lazy
    def __init__(self, pandas_obj: pd.DataFrame):
        pandas_obj = self._format(pandas_obj)
        self._obj = pandas_obj

    @classmethod
    def _format(cls, pandas_obj: pd.DataFrame) -> pd.DataFrame:
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

    # TODO plot durations using plotly
    # TODO format and plot other dfs on rows
    def plot(self, *dfs):
        # TODO raise if plotly not installed
        pass

    def union(self, *dfs) -> pd.DataFrame:
        return intervals_union(
            [self._obj, *[self._format(df) for df in dfs]],
            sort_cols=self.additional_cols,
        )

    def intersection(self, *dfs) -> pd.DataFrame:
        return intervals_intersection(
            [self._obj, *[self._format(df) for df in dfs]],
            groupby_cols=self.groupby_cols,
        )

    def overlap(self, *dfs) -> pd.DataFrame:
        return intervals_overlap(
            [self._obj, *[self._format(df) for df in dfs]],
            groupby_cols=self.groupby_cols,
        )

    def non_overlap(self, *dfs) -> pd.DataFrame:
        return intervals_non_overlap(
            [self._obj, *[self._format(df) for df in dfs]],
            groupby_cols=self.groupby_cols,
        )

    def combine(self, *dfs) -> pd.DataFrame:
        return intervals_combine(
            [self._obj, *[self._format(df) for df in dfs]],
            groupby_cols=self.groupby_cols,
            aggregations=self.aggregations,
        )

    def pad(
        self,
        pad: Optional[float] = None,
        left_pad: Optional[float] = None,
        right_pad: Optional[float] = None,
    ):
        if pad is not None:
            if left_pad is not None or right_pad is not None:
                raise ValueError(
                    "Either use `pad`, or `left_pad`/`right_pad`."
                )
            left_pad, right_pad = pad, pad
        self._obj["start"] = self._obj["start"] - left_pad
        self._obj["end"] = self._obj["end"] + right_pad
        return self._obj

    def unpad(
        self,
        unpad: Optional[float] = None,
        left_unpad: Optional[float] = None,
        right_unpad: Optional[float] = None,
    ):
        if unpad is not None:
            if left_unpad is not None or right_unpad is not None:
                raise ValueError(
                    "Either use `unpad`, or `left_unpad`/`right_unpad`."
                )
            left_unpad, right_unpad = unpad, unpad

        self._obj["start"] = self._obj["start"] + left_unpad
        self._obj["end"] = self._obj["end"] - right_unpad
        return self._obj.loc[self._obj["end"] - self._obj["start"] >= 0]

    def diff(self, *dfs):
        return intervals_difference(
            self._obj,
            [self._format(df) for df in dfs],
            groupby_cols=self.groupby_cols,
        )

    # TODO complement (w configurable endpoints)
    def complement(
        self, left_bound: Optional[float] = None, right_bound: Optional[float] = None
    ):
        return intervals_complement(
            self._obj,
            groupby_cols=self.groupby_cols,
        )
        # return intervals_difference(
        #     [self._obj, *[self._format(df) for df in dfs]],
        #     groupby_cols=self.groupby_cols,
        # )


def intervals_union(
    dfs: List[pd.DataFrame], sort_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    intervals = pd.concat(dfs, axis=0)
    if intervals.empty:
        return intervals
    sort_cols = sort_cols if sort_cols is not None else []
    return intervals.sort_values(["start", "end", *sort_cols]).drop_duplicates()

# TODO searchsorted implementation
def _get_overlapping_mask(df: pd.DataFrame) -> np.ndarray:
    df = df.sort_values("start")
    intervals_a = df.iloc[:, :2].values

    mask = []
    for start_a, end_a in intervals_a:
        overlap = False
        for start_b, end_b in intervals_a:
            if (
                (start_b < start_a < end_b < end_a)
                or (start_a < start_b < end_a < end_b)
                or ((start_a < start_b) and (end_b < end_a))
            ):
                overlap = True
                break
        mask.append(overlap)
    return np.array(mask)

# def _get_overlapping_mask(df: pd.DataFrame) -> np.ndarray:
#     df = df.sort_values("start")
#     starts, ends = df["start"].values, df["end"].values
#     overlaps = starts[1:] - ends[:-1]
#     mask = (overlaps < 0)
#     mask = np.append(mask, mask[-1])
#     return mask

def intervals_overlap(
    dfs: List[pd.DataFrame],
    groupby_cols: Optional[List[str]] = None,
):
    intervals = pd.concat(dfs, axis=0)
    if intervals.empty:
        return intervals

    if groupby_cols is None:
        groupby_cols = []

    if len(groupby_cols) == 0:
        mask = _get_overlapping_mask(intervals)
        return intervals.loc[mask]

    results = []
    for _, df_group in intervals.groupby(groupby_cols):
        mask = _get_overlapping_mask(df_group)
        result =  df_group.loc[mask]
        results.append(result)
    return pd.concat(results, axis=0).sort_values("start")

# TODO reduce duplication
def intervals_non_overlap(
    dfs: List[pd.DataFrame],
    groupby_cols: Optional[List[str]] = None,
):
    intervals = pd.concat(dfs, axis=0)
    if intervals.empty:
        return intervals

    if groupby_cols is None:
        groupby_cols = []

    if len(groupby_cols) == 0:
        mask = _get_overlapping_mask(intervals)
        return intervals.loc[~mask]

    results = []
    for _, df_group in intervals.groupby(groupby_cols):
        mask = _get_overlapping_mask(df_group)
        result =  df_group.loc[~mask]
        results.append(result)
    return pd.concat(results, axis=0).sort_values("start")

def intervals_combine(
    dfs: List[pd.DataFrame],
    groupby_cols: Optional[List[str]] = None,
    aggregations: Optional[Dict[str, Union[str, Callable]]] = None,
):
    intervals = intervals_union(dfs, groupby_cols)
    if intervals.empty:
        return intervals

    if aggregations is None:
        aggregations = {}

    aggregations = {c: "first" for c in intervals.columns if c not in aggregations}

    combined_labels = []
    for _, interval_group in intervals.groupby(groupby_cols, as_index=False):
        interval_group_sorted = interval_group.sort_values("start")

        # TODO Vectorise this
        # Loop over labels and compare each to the previous label to find labels to combine
        group_inds = []
        ind, interval_end_time = 0, 0
        for start, end in interval_group_sorted[["start", "end"]].values:
            # If interval is within previous label, combine them
            if start <= interval_end_time:
                interval_end_time = max(interval_end_time, end)
                group_inds.append(ind)
            # If not, start a new interval
            else:
                interval_end_time = end
                ind += 1
                group_inds.append(ind)

        grpd_labels = interval_group_sorted.groupby(group_inds).agg(aggregations)
        combined_labels.append(grpd_labels)

    return pd.concat(combined_labels).reset_index(drop=True)

# TODO Remove duplication
def _intervals_overlapping(intervals: Union[np.ndarray, pd.DataFrame]):
    if isinstance(intervals, pd.DataFrame):
        intervals = intervals[["start", "end"]].values
    intervals = intervals[np.argsort(intervals[:, 0]), :]
    starts, ends = intervals[:, 0], intervals[:, 1]
    overlaps = starts[1:] - ends[:-1]
    return (overlaps < 0).any()

def _points_from_intervals(interval_groups: List[np.ndarray]) -> Tuple[np.ndarray]:
    n_interval_groups = len(interval_groups)
    interval_points, interval_indices = [], []
    for i, intervals in enumerate(interval_groups):
        assert not _intervals_overlapping(
            intervals
        ), "Expected the intervals within a group to be non-overlapping"
        n_intervals = len(intervals)

        indices = np.zeros((n_intervals, n_interval_groups))
        indices[:, i] = np.arange(n_intervals) + 1
        indices = np.concatenate([indices, -indices], axis=0)

        points = np.concatenate([intervals[:, 0:1], intervals[:, 1:2]], axis=0)

        interval_points.append(points)
        interval_indices.append(indices)

    interval_points = np.concatenate(interval_points, axis=0)
    interval_indices = np.concatenate(interval_indices, axis=0)

    idx = np.argsort(interval_points[:, 0])
    interval_points = interval_points[idx, :]
    interval_indices = interval_indices[idx, :]

    interval_indices = np.abs(np.cumsum(interval_indices, axis=0)) - 1
    return interval_points, interval_indices

def _atomize_intervals(
    interval_groups: List[np.ndarray],
    min_len: Optional[float] = None,
    drop_gaps: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    points, indices = _points_from_intervals(interval_groups)
    for i in range(1, len(interval_groups)):
        indices[indices[:, i] != -1, :i] = -1

    starts, ends = points[:-1, 0:1], points[1:, 0:1]
    interval_idxs = indices[:-1].astype(int)
    atomized_intervals = np.concatenate([starts, ends], axis=1)

    if drop_gaps:
        mask_nongap_intervals = (interval_idxs != -1).any(axis=1)

        atomized_intervals = atomized_intervals[mask_nongap_intervals]
        interval_idxs = interval_idxs[mask_nongap_intervals]

    if min_len is not None:
        interval_lengths = atomized_intervals[:, 1] - atomized_intervals[:, 0]
        mask_above_min_len = interval_lengths > min_len

        atomized_intervals = atomized_intervals[mask_above_min_len]
        interval_idxs = interval_idxs[mask_above_min_len]

    return atomized_intervals, interval_idxs

def intervals_difference(
    df: pd.DataFrame,
    dfs: List[pd.DataFrame],
    groupby_cols: Optional[List[str]] = None,
    min_len: Optional[float] = None,
):
    if len(dfs) == 0:
        return df

    intervals_b = intervals_combine(dfs, groupby_cols)
    if len(intervals_b) == 0:
        return df

    intervals_a = intervals_combine([df], groupby_cols)

    # TODO Fix groupby to select vals in intervals_b
    results = []
    for _, intervals_a_group in intervals_a.groupby(groupby_cols):
        input_columns = df.columns
        intervals_a_metadata = df.drop(["start", "end"], axis=1)

        intervals_a = intervals_a[["start", "end"]].values.copy()
        intervals_b = intervals_b[["start", "end"]].values.copy()

        atoms, indices = _atomize_intervals(
            [intervals_a, intervals_b],
            drop_gaps=False,
            min_len=min_len,
        )
        mask_a_atoms = (indices[:, 0] != -1) & (indices[:, 1] == -1)
        result, indices = atoms[mask_a_atoms], indices[mask_a_atoms, 0]

        intervals_a_diff_b = intervals_a_metadata.iloc[indices].reset_index(drop=True)
        intervals_a_diff_b[["start", "end"]] = result
        results.append(intervals_a_diff_b[input_columns])

    return intervals_union(result, groupby_cols)


def intervals_complement(
    df: pd.DataFrame,
    groupby_cols: Optional[List[str]] = None,
):
    df = intervals_combine(df, groupby_cols)
    result = []
    for vals, df_group in df.groupby(groupby_cols):
        ...
        # TODO append and prepend zero duration labels
        # TODO get starts and ends
        # TODO Append to results
    return intervals_union(result, groupby_cols)

