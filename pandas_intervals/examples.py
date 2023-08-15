"""Examples of a few IntervalsFrame subclasses"""
import json
from os import PathLike

from pandas_intervals.intervals_accessor import IntervalsAccessor
from pandas_intervals.utils import comma_join

import pandas as pd


@pd.api.extensions.register_dataframe_accessor("reg")  # Name of new accessor, pd.DataFrame.<name>
class RegionsAccessor(IntervalsAccessor):
    # Additional required columns can be specified in a list of tuple
    # where each tuple is `(column_name, dtype, aggregation)`
    additional_cols = [
        ("tag", "int64", "groupby"),
        ("note", "object", lambda x: ",".join(x)),
    ]

    # Default values for columns can be specified as a dictionary,
    # columns that don't appear in this list are assumed to be necessary
    default_values = {
        "note": "",
    }

    # Add whatever methods/properties you want!
    @property
    def all_notes(self):
        return self.df["note"]

    def concat_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.format(df)
        return pd.concat([self.df, df], axis=0).sort_values("tag")

    # Create constructors to parse and validate data from other sources
    @classmethod
    def from_json(cls, filepath: PathLike) -> pd.DataFrame:
        with open(filepath, 'r', encoding='utf-8') as f:
            intervals = json.load(f)

        assert isinstance(intervals, list)
        assert len(intervals) > 0
        assert isinstance(intervals[0], dict)
        assert all(k in intervals[0] for k in ["start", "end", "tag", "note"])

        df = pd.DataFrame(intervals)
        return cls.format(df)
