# pandas-intervals
Pandas extension for `DataFrame`s of intervals (AKA `spandas`).

NOTE: Still very much a WIP!!


This library provides `IntervalsFrames`: a standard interface for `DataFrames` of "interval"-like objects, where each object is specified by a `start` and an `end`. `IntervalsFrames` are `DataFrame`-like, but with some extras:
* The columns `"start"` and `"end"` are automatically formatted as `float` types
* The order of the columns is strictly as specified
* Interval set operations (intersections, combinations, differences) are implemented via magic methods

The `IntervalsFrame` object is also extensible and allows adding columns with default values and specifying how columns are aggregated when combining close intervals.

## Quick start
Say we have two sets of intervals `A` and `B` as specified by:
```
  A:     (----]    (----]         (--------------]
  B: (--]   (---]          (----]    (---]    (------] (---]
```

We can represent these sets in `python` using `DataFrames`, and
```python
import pandas as pd
import pandas_intervals

vf_a = pd.DataFrame(
    [
        [50, 100],
        [150, 200],
        [300, 450],
    ],
).ivl()
print(vf_a)
#    start    end
# 0   50.0  100.0
# 1  150.0  200.0
# 2  300.0  450.0

vf_b = pd.DataFrame(
    [
        [10, 40],
        [80, 120],
        [230, 280],
        [330, 370],
        [420, 490],
        [510, 550],
    ],
).ivl()
print(vf_b)
#    start    end
# 0   10.0   40.0
# 1   80.0  120.0
# 2  230.0  280.0
# 3  330.0  370.0
# 4  420.0  490.0
# 5  510.0  550.0
```

We have all the standard methods available to DataFrames, but we also now have native interval set operations implemented trhough the `ivl` accessor:
```python
union = vf_a.ivl.union(vf_b)

intersection = vf_a.ivl.intersection(vf_b)

combined = vf_a.ivl.combine(vf_b)

padded = vf_a.ivl.pad(10)  # Optional kwargs: `left_pad`, `right_pad`

unpadded = vf_a.ivl.unpad(10)  # Optional kwargs: `left_unpad`, `right_unpad`

diff = vf_a.ivl.diff(vf_b)

complement = vf_a.ivl.complement()  # Optional kwargs: `left_bound`, `right_bound`
```

We can also plot these intervals using a plotly backend (if available):
```python
vf_a.ivl.plot()  # Plot a single DataFrame of intervals
vf_a.ivl.plot(vf_b)  # Compare multiple DataFrames of intervals on a plot
```

This interface can easily be extended, we can add additional columns with default values and types.
For example, if we want to create an intervals accessor called `"regions"` which
* Has 2 extra columns ("tag" and "note"),
* Column "tag" must be specified, but "note" is optional,
* Column "tag" is an integer, and "note" is a string,
* Aggregations are done across different values of "tag", and "note" values are combined
    into a comma-separated string.

We can accomplish this in a relatively small class:

```python
@pd.api.extensions.register_dataframe_accessor("regions")  # Name of new accessor
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
```

After defining and running this snippet, we now have
* `pd.DataFrame(..).regions.all_notes()` available as a method on any DataFrame,
* `pd.DataFrame(..).regions()` will return a formatted DataFrame as specified by the fields in `RegionsAccessor.additional_cols`,
* All the methods on the `ivl` accessor come along for free!
