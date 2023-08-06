from __future__ import annotations

from typing import Callable, Union, List, Dict, Type
from numbers import Number

import numpy as np
import pandas as pd


# TODO Impl is_formatted method
# TODO Explore __init__ override, see GeoPandas for example (constructor from_df)
# https://github.com/geopandas/geopandas/blob/main/geopandas/geodataframe.py#L136
# https://github.com/pandas-dev/pandas/blob/v1.5.0/pandas/core/frame.py#L608
class IntervalsFrame(pd.DataFrame):
    _required_fields = [("start", float, "min"), ("end", float, "max")]
    additional_fields = []

    @property
    def _constructor(self) -> Type:
        """Return an IntervalsFrame type when a Pandas operation creates a new instance.
        https://pandas.pydata.org/docs/development/extending.html#override-constructor-properties
        """
        return type(self)

    @classmethod
    @property
    def fields(cls) -> Dict[str, Union[str, Callable]]:
        return [*cls._required_fields, *cls.additional_fields]

    @classmethod
    @property
    def required_cols(cls) -> List[str]:
        return [i[0] for i in cls._required_fields]

    @classmethod
    @property
    def additional_cols(cls) -> List[str]:
        return [i[0] for i in cls.additional_fields]

    @classmethod
    @property
    def aggregations(cls) -> Dict[str, Union[str, Callable]]:
        return {col: agg for col, _, agg in cls.fields if agg != "groupby"}

    @classmethod
    @property
    def groupby_cols(cls) -> List[str]:
        return [col for col, _, agg in cls.fields if agg == "groupby"]

    def format(self) -> IntervalsFrame:
        """Ensure the IntervalsFrame has expected columns and dtypes."""
        if self.columns.empty:  # Object is empty (no columns even specified)
            return self.empty_frame()

        labels = self.copy()
        labels = labels.rename(
            columns={
                i: col
                for i, col in enumerate([*self.required_cols, *self.additional_cols])
            }
        )
        missing_cols = [col for col in self.required_cols if col not in labels]
        if len(missing_cols) > 0:
            raise ValueError(
                f"LabelsFrame missing required column(s) '{', '.join(missing_cols)}'"
            )

        # Set default columns and fill missing values
        for col, default_val, _ in self.additional_fields:
            if col not in labels:
                labels[col] = default_val
            else:
                labels[col] = labels[col].fillna(default_val)

        # Format numeric cols
        numeric_cols = ["start", "end"]
        if "confidence" in labels:
            numeric_cols += ["confidence"]
        labels[numeric_cols] = labels[numeric_cols].astype(float)

        # Sort in standard order
        cols_to_keep = self.required_cols + [i[0] for i in self.additional_fields]
        return labels[cols_to_keep]

    @classmethod
    def empty_frame(cls) -> IntervalsFrame:
        dtypes = np.dtype(
            [(col_name, col_type) for col_name, col_type, _ in cls._required_fields]
        )
        return cls(np.empty(0, dtype=dtypes)).format()

    def __or__(self, other):
        if not isinstance(other, IntervalsFrame):
            return super().__or__(other)
        return intervals_union(self, other)

    # TODO searchsorted implementation
    def _get_overlapping(self) -> np.ndarray:
        self = self.sort_values("start")
        starts, ends = self["start"].values, self["end"].values
        overlaps = starts[1:] - ends[:-1]
        self = self.iloc[:-1]
        return self.loc[(overlaps < 0)]

    # TODO Make sure overlaps within self and other are taken care of (combine close before?)
    def __and__(self, other):
        if not isinstance(other, IntervalsFrame):
            return super().__and__(other)

        intervals = pd.concat([self, other], axis=0)

        cols_groupby = [
            col for col, _, agg in self.additional_fields if agg == "groupby"
        ]
        if len(cols_groupby) == 0:
            return self._get_overlapping()

        result = []
        for _, ivf in intervals.groupby(cols_groupby):
            result.append(ivf._get_overlapping())
        return pd.concat(result, axis=0).sort_values("start")

    def _combine_overlapping(self, data: pd.DataFrame) -> pd.DataFrame:
        labels = data.copy()
        if labels.empty:
            return labels

        # Define aggregations for combining labels
        aggregations = {c: "first" for c in labels.columns}
        aggregations.update(self.aggregations)

        combined_labels = []
        group_by = self.groupby_cols
        for _, tag_labels in labels.groupby(group_by, as_index=False):
            tag_labels = tag_labels.sort_values("start")

            # TODO Vectorise this
            # Loop over labels and compare each to the previous label to find labels to combine
            group_inds = []
            ind, label_end_time = 0, 0
            for start, end in tag_labels[["start", "end"]].values:
                # If label is within `gap_size` of the previous label, combine them
                if start <= label_end_time:
                    label_end_time = max(label_end_time, end)
                    group_inds.append(ind)
                # If not, start a new label
                else:
                    label_end_time = end
                    ind += 1
                    group_inds.append(ind)

            grpd_labels = tag_labels.groupby(group_inds).agg(aggregations)
            combined_labels.append(grpd_labels)

        return self.__class__(pd.concat(combined_labels).reset_index(drop=True))

    def __add__(self, other):
        if not isinstance(other, IntervalsFrame):
            if isinstance(other, Number):
                self["start"] -= other
                self["end"] += other
                return self
            return super().__add__(other)
        return self._combine_overlapping(pd.concat([self, other]))

    def __pos__(self):
        return self._combine_overlapping(self)

    # TODO Interval subtraction (IntervalDiff)
    def __sub__(self, other):
        if not isinstance(other, IntervalsFrame):
            if isinstance(other, Number):
                self["start"] += other
                self["end"] -= other
                return self.loc[self["end"] - self["start"] >= 0]
            return super().__sub__(other)
        raise NotImplementedError("SUB")

    # TODO Interval complement (??)
    def __invert__(self):
        raise NotImplementedError("INVERT")


def intervals_union(vf_a: IntervalsFrame, vf_b: IntervalsFrame) -> IntervalsFrame:
    return (
        pd.concat([vf_a, vf_b], axis=0)
        .sort_values(["start", *vf_a.additional_cols])
        .drop_duplicates()
    )
