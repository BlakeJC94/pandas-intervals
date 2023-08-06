"""Examples of a few IntervalsFrame subclasses"""
from __future__ import annotations

from pandas_intervals.core import IntervalsFrame
from pandas_intervals.utils import comma_join


class LabelsFrame(IntervalsFrame):
    additional_fields = [
        ("tag", "undefined", "groupby"),
        ("study_id", "", "groupby"),
        ("confidence", 0.0, "max"),
        ("note", "", comma_join),
    ]


class RegionsFrame(IntervalsFrame):
    additional_fields = [
        ("study_id", "", "groupby"),
        ("category", "", "groupby"),
    ]
