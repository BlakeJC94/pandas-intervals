"""Examples of a few IntervalsFrame subclasses"""

from pandas_intervals.intervals_accessor import IntervalsAccessor
from pandas_intervals.utils import comma_join

import pandas as pd


@pd.api.extensions.register_dataframe_accessor("reg")  # Name of new accessor
class RegionsAccessor(IntervalsAccessor):

    # Additional required columns can be specified in a list of tuple
    # where each tuple is `(column_name, dtype, aggregation)`
    additional_cols = [
        ("tag", "int64", "groupby"),
        ("note", "object", lambda x: ','.join(x)),
    ]

    # Default values for columns can be specified as a dictionary,
    # columns that don't appear in this list are assumed to be necessary
    default_values = {
         "note": "",
    }

    # Add whatever methods/properties you want!
    def all_notes(self):
        return self._obj["note"]

    # TODO Add a constructor example
