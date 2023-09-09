# pandas-intervals
Pandas extension for `DataFrame`s of intervals (AKA `spandas`).

TODO:
* [x] Build up bash DataFrame accessor
* [x] Structure GroupBys to apply interval functions on specific groups of intervals
* [x] Implement iterative methods for interval operations
    * [x] Implement unit tests for iterative methods
    * [x] Add more edge cases to the iterative methods
* [x] Implement plot method for accessor
* [x] Vectorise all methods
    * [x] Implement property-based tests
    * [x] Refactor tests to test specific aspects
* [x] Set difference
* [x] Acceptance tests
* [x] Upgrade asserter
* [x] Documentation
* [x] Doctests
* [ ] Decorator


This library provides `ivl` extension: a standard interface and methods for `DataFrames` of "interval"-like objects, where each object is specified by a `start` and an `end`.
* The columns `"start"` and `"end"` are automatically formatted as `float` types,
* The order of the columns is strictly as specified,
* Interval set operations (intersections, combinations, differences) can be called via the `ivl` property on any DataFrame after importing this module.

The `IntervalsAccessor` object is also extensible and allows adding columns with default values and specifying how columns are aggregated when combining close intervals.

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

df_a = pd.DataFrame(
    [
        (50, 100),
        (150, 200),
        (300, 450),
    ],
).ivl()
print(df_a)
#    start    end
# 0   50.0  100.0
# 1  150.0  200.0
# 2  300.0  450.0

df_b = pd.DataFrame(
    [
        (10, 40),
        (80, 120),
        (230, 280),
        (330, 370),
        (420, 490),
        (510, 550),
    ],
).ivl()
print(df_b)
#    start    end
# 0   10.0   40.0
# 1   80.0  120.0
# 2  230.0  280.0
# 3  330.0  370.0
# 4  420.0  490.0
# 5  510.0  550.0
```

We have all the standard methods available to DataFrames, but we also now have native interval set operations implemented through the `ivl` accessor:
```python
union = df_a.ivl.union(df_b)

intersection = df_a.ivl.intersection(df_b)

combined = df_a.ivl.combine(df_b)

padded = df_a.ivl.pad(10)  # Optional kwargs: `left_pad`, `right_pad`

trunc = df_a.ivl.truncate(df_b)

complement = df_a.ivl.complement()  # Optional kwargs: `left_bound`, `right_bound`

df_a_contains_df_b = df_a.ivl.contains(df_b)

df_a_min_dist_to_b = df_a.ivl.nearest(df_b)
```

We can also plot intervals using a Plotly backend (if available):
```python
df_a.ivl.plot()
```
![image](https://github.com/BlakeJC94/pandas-intervals/assets/16640474/4133ac9c-def5-4a4a-8cc8-d17badf9c054)



Multiple groups of intervals can also be plotted on the same graph as well:
```python
results = []
for df in [df_a, df_b, df_c]:
    df['group'] = i
    results.append(df)

results = pd.concat(results)
results.ivl.plot(groupby_cols=['group'])
```
![image](https://github.com/BlakeJC94/pandas-intervals/assets/16640474/cad158ca-042b-4878-9377-639094ece0d8)


## Extensions

This interface can easily be extended, we can add additional columns with default values and types.
For example, if we want to create an intervals accessor called `"regions"` which
* Has 2 extra columns ("tag" and "note"),
* Column "tag" must be specified, but "note" is optional,
* Column "tag" is an integer, and "note" is a string,
* Aggregations are done across different values of "tag", and "note" values are combined
    into a comma-separated string.

We can accomplish this in a relatively small class:

```python
import json
from os import PathLike
from typing import List

import pandas as pd
from pandas_intervals import IntervalsAccessor


# Let's define a function to be used when aggregating results across the `notes` column
def comma_join(x: List[str]) -> str:
    return ", ".join(sorted({i.strip() for n in x if n for i in n.split(",")}))


@pd.api.extensions.register_dataframe_accessor("reg")  # Name of new accessor, pd.DataFrame.<name>
class RegionsAccessor(IntervalsAccessor):
    # Additional required columns can be specified in a list of tuple
    # where each tuple is `(column_name, dtype, aggregation)`
    additional_fields = [
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
```

After defining and running this snippet, we now have
* `pd.DataFrame(..).reg.all_notes()` available as a method on any DataFrame,
* `pd.DataFrame(..).reg()` will return a formatted DataFrame as specified by the fields in `RegionsAccessor.additional_cols`,
* All the methods on the `ivl` accessor come along for free!
