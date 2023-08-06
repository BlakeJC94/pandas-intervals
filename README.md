# pandas-intervals
Pandas extension for `DataFrame`s of intervals (AKA `spandas`).

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

We can represent these sets in `python` using `IntervalsFrames`
```python
from pandas_intervals import IntervalsFrame

vf_a = IntervalsFrame(
    [
        [50, 100],
        [150, 200],
        [300, 450],
    ],
)
print(sf_a)

vf_b = IntervalsFrame(
    [
        [10, 40],
        [80, 120],
        [230, 280],
        [330, 370],
        [420, 490],
        [510, 550],
    ],
)
print(sf_b)
```

We have all the standard methods available to DataFrames:
```python
import pandas as pd
print(f"{isinstance(vf_a, pd.DataFrame) = }")

for i, vf in vf_b.groupby(vf_a.index // 2):
    print(f"--- group {i}:")
    print(vf)
```

But we also have native interval set methods implemented as magic methods:
```python
union = vf_a | vf_b

intersection = vf_a & vf_b

combined = vf_a + vf_b

padded = vf_a + 10

unpadded = vf_a - 10

# diff = vf_a - vf_b  # TODO

# complement = ~vf_a  # TODO
```


This interface can easily be extended, we can add additional columns with default values and types

```python
class RegionsFrame(IntervalsFrame):
    # Additional columns to be specified as a list of `(name, value, aggregation)` tuples
    additional_fields = [
        ("tag", "undefined", "first"),
        ("note", "", lambda x: ','.join(x)),
    ]
```
