from typing import List, Dict, Optional, List, Any
from typing_extensions import Self

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
    diff,
    symdiff,
    complement,
    combine,
    truncate,
    nearest,
)
from .vis import plot_intervals
from .utils import sort_intervals, FieldsTrait, FormatTrait, Field


Figure = plotly.graph_objects.Figure if plotly is not None else None


class IntervalsAccessor(FieldsTrait, FormatTrait):
    """A DataFrame accessor for frames containing intervals (columns "start" and "end").

    Invoking this accessor on a DataFrame will check the frame is a valid representation of
    intervals, set types on the columns if needed, and sort the columns in a standard order so that
    column 0 and 1 are "start" and "end".

    To simply validate and format a DataFrame containing intervals, call this accessor on a
    DataFrame:

        >>> df = pd.DataFrame([(100, 200), (300, 400)])
        >>> print(df)
             0    1
        0  100  200
        1  300  400
        >>> df = df.ivl()
        >>> print(df)
           start    end
        0  100.0  200.0
        1  300.0  400.0

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
        ...     additional_fields = [
        ...         ("tag", "int64", "groupby"),
        ...         ("note", "object", lambda x: ','.join(x)),
        ...     ]
        ...
        ...     # Default values for columns can be specified as a dictionary,
        ...     # columns that don't appear in this list are assumed to be necessary
        ...     default_values = {
        ...          "note": "",
        ...     }

    Attributes:
        df: DataFrame that's accessed.
        additional_fields: List of 3-tuples for additional columns to manage in accessor. Each
            element is a tuple of name, type, and aggregator. The aggregator must match the spec
            provided by `pd.DataFrame.groupby.agg(..)`, which can be a string (name of basic
            operation) or a callable that maps a list of values to a single value. The reserved
            aggregation `"groupby"` is used to group sets of intervals across additional column
            values to apply operations without overlap.
        default_values: Dictionary of `additional_fields` columns names to default values to fill in
            for those columns when formatting a DataFrame. An error will be raised if an additional
            column has not been given a default value here.
    """

    additional_fields: List[Field] = []
    default_values: Dict[str, Any] = {}

    def __init__(self, pandas_obj: Optional[pd.DataFrame] = None):
        """Initialise the accessor. Called automatically when using `pd.DataFrame(...).ivl` or
        `pd.DataFrame.ivl`.

        When the accessor is used, the `__init__` method will call the `format` method defined to
        * Check the accessor has a valid configuration,
        * Check DataFrame has required columns,
        * Add additional cols with configured defaults,
        * Check types and change if needed.

        This will ensure that all the operations on DataFrames of interval-like objects have controlled
        types and column names and orders across all operations used throughout this extension.

        Args:
            pandas_obj: Object where this accessor is called from.
        """
        self.df = self.format(pandas_obj)
        self._basic = False

    def __call__(self) -> pd.DataFrame:
        """Return the formatted DataFrame."""
        return self.df

    @property
    def durations(self) -> pd.Series:
        """Return the durations of the intervals.

        >>> df = pd.DataFrame([(100, 200), (300, 500)])
        >>> df.ivl.durations
        0    100.0
        1    200.0
        dtype: float64

        """
        starts = self.df["start"]
        ends = self.df["end"]
        return ends - starts

    @property
    def span(self) -> float:
        """Return the total time covered in the intervals DataFrame.

        >>> df = pd.DataFrame([(100, 200), (300, 500)])
        >>> df.ivl.span
        400.0

        """
        return self.df["end"].max() - self.df["start"].min()

    @property
    def basic(self) -> Self:
        """Toggle usage of basic (non-vectorised) algorithms.

        >>> df = pd.DataFrame([(100, 200), (150, 300), (400, 500)], columns=['start', 'end'])
        >>> df.ivl.overlap()
           start    end
        0  100.0  200.0
        1  150.0  300.0
        >>> df.ivl.basic.overlap()
           start    end
        0  100.0  200.0
        1  150.0  300.0

        """
        self._basic = True
        return self

    def plot(
        self,
        groupby_cols: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        **layout_kwargs,
    ) -> Figure:
        """Plots a DataFrame of intervals. Plotly must be available for this to work.

            >>> df = pd.DataFrame([(100, 200), (400, 500)], columns=['start', 'end'])
            >>> df.ivl.plot()  # doctest: +SKIP

        Args:
            groupby_cols: List of columns to group DataFrame by before plotting. By default uses the
                `additional_cols` with a `"groupy"` agg value.
            colors: List of plotly colors to use when plotting intervals.
            layout_kwargs: Kwargs to pass to the `plotly.go.Figure.layout` method.
        """
        if plotly is None:
            raise ImportError("Plotting intervals requires `plotly` to be installed")

        groupby_cols = groupby_cols or self.groupby_cols
        return plot_intervals(self.df, groupby_cols, colors, **layout_kwargs)

    def sort(self) -> pd.DataFrame:
        """Sort DataFrame by 'start', 'end', and each additional column in order.

        >>> df = pd.DataFrame([(400, 450), (400, 500), (100, 200)], columns=['start', 'end'])
        >>> df.ivl.sort()
           start    end
        2  100.0  200.0
        0  400.0  450.0
        1  400.0  500.0

        """
        results = sort_intervals(
            self.df,
            sort_cols=self.additional_cols,
        )
        return results

    # TODO Test unit
    def shorter_than(self, upper_bound: float, strict: bool = False) -> pd.DataFrame:
        """Filters DataFrames to rows containing intervals that are shorter than a given duration.

            >>> df = pd.DataFrame([(400, 450), (400, 500), (100, 200)], columns=['start', 'end'])
            >>> df.ivl.shorter_than(80)
               start    end
            0  400.0  450.0

        Args:
            upper_bound: Maximum allowable duration to return.
            strict: Whether to use a strict inequality (False by default).
        """
        if strict:
            return self.df.loc[self.durations < float(upper_bound)]
        else:
            return self.df.loc[self.durations <= float(upper_bound)]

    # TODO Test unit
    def longer_than(self, lower_bound: float, strict: bool = False) -> pd.DataFrame:
        """Filters DataFrames to rows containing intervals that are longer than a given duration.

            >>> df = pd.DataFrame([(400, 450), (400, 500), (100, 200)], columns=['start', 'end'])
            >>> df.ivl.longer_than(80)
               start    end
            1  400.0  500.0
            2  100.0  200.0

        Args:
            upper_bound: Maximum allowable duration to return.
            strict: Whether to use a strict inequality (False by default).
        """
        if strict:
            return self.df.loc[self.durations > float(lower_bound)]
        else:
            return self.df.loc[self.durations >= float(lower_bound)]

    # TODO Test unit
    def between(
        self,
        left_bound: Optional[float] = None,
        right_bound: Optional[float] = None,
        strict: bool = False,
    ) -> pd.DataFrame:
        """Filters DataFrames to rows containing intervals that are between two given time points.

            >>> df = pd.DataFrame([(100, 160), (200, 250), (400, 500)], columns=['start', 'end'])
            >>> df.ivl.between(150, 300)
               start    end
            0  100.0  160.0
            1  200.0  250.0

        Args:
            left_bound: Left endpoint of bounds.
            right_bound: Right endpoint of bounds.
            strict: Whether to return intervals completely within the bounds or return intervals
                with any overlap with bounds (False by default).
        """
        if left_bound is None:
            left_bound = self.df["start"].min()
        if right_bound is None:
            left_bound = self.df["end"].max()

        assert left_bound <= right_bound

        if not strict:
            mask = (self.df["end"] >= left_bound) & (self.df["start"] <= right_bound)
        else:
            mask = (self.df["start"] >= left_bound) & (self.df["end"] <= right_bound)

        return self.df.loc[mask]

    # TODO Test unit
    def contains(self, df: pd.DataFrame) -> bool:
        """Returns whether a DataFrame contains all intervals found in another DataFrame.

            >>> df_a = pd.DataFrame([(100, 160), (200, 250), (400, 500)], columns=['start', 'end'])
            >>> df_b = pd.DataFrame([(100, 160)], columns=['start', 'end'])
            >>> df_c = pd.DataFrame([(700, 860)], columns=['start', 'end'])
            >>> df_a.ivl.contains(df_b)
            True
            >>> df_b.ivl.contains(df_a)
            False
            >>> df_a.ivl.contains(df_c)
            False

        Args:
            df: Another DataFrame matching the Intervals spec.
        """
        df = df.drop_duplicates()
        df_all = self.union(df)
        return len(self.df.drop_duplicates()) == len(df_all)

    def pad(
        self,
        pad: Optional[float] = None,
        left_pad: Optional[float] = None,
        right_pad: Optional[float] = None,
    ):
        """Pads the intervals in the DataFrame by a specified amount.

            >>> df = pd.DataFrame([(100, 160), (200, 250)], columns=['start', 'end'])
            >>> df.ivl.pad(20)
               start    end
            0   80.0  180.0
            1  180.0  270.0
            >>> df.ivl.pad(left_pad=30)
               start    end
            0   70.0  160.0
            1  170.0  250.0
            >>> df.ivl.pad(right_pad=-55)
               start    end
            0  100.0  105.0

        Args:
            pad: Value to pad both 'start' and 'end' columns with.
            left_pad: Value to pad only 'start' column with.
            right_pad: Value to pad only 'end' column with.
        """
        if pad is not None:
            if left_pad is not None or right_pad is not None:
                raise ValueError("Either use `pad`, or `left_pad`/`right_pad`.")
            left_pad, right_pad = pad, pad

        starts = self.df["start"] - (left_pad or 0)
        ends = self.df["end"] + (right_pad or 0)
        mask = ends - starts >= 0
        return self.df.assign(start=starts, end=ends).loc[mask]

    def union(self, *dfs) -> pd.DataFrame:
        """Return the union with one or more DataFrames of intervals.

        Example:
        ```
        #     A  :   [----)     [----)
        #     B  :   [----)             [----)
        #
        # Result :   [----)     [----)  [----)
        ```

            >>> df_a = pd.DataFrame([(100, 160), (200, 250)], columns=['start', 'end'])
            >>> df_b = pd.DataFrame([(100, 160), (400, 550)], columns=['start', 'end'])
            >>> df_a.ivl.union(df_b)
               start    end
            0  100.0  160.0
            1  200.0  250.0
            1  400.0  550.0

        Args:
            dfs: DataFrames of intervals to take union with.
        """
        interval_sets = [self.df, *[self.format(df) for df in dfs]]
        return pd.concat(interval_sets, axis=0).drop_duplicates()

    def overlap(self) -> pd.DataFrame:
        """Return the intervals in a DataFrame which overlap with each other.

        Example:
        ```
        #     A  :   [----)     [----)
        #               [----)
        #
        # Result :   [----)
        #               [----)
        ```

            >>> df = pd.DataFrame([(100, 220), (200, 250), (400, 500)], columns=['start', 'end'])
            >>> df.ivl.overlap()
               start    end
            0  100.0  220.0
            1  200.0  250.0

        Compatible with `basic` flag.

            >>> df.ivl.basic.overlap()
               start    end
            0  100.0  220.0
            1  200.0  250.0

        """
        operation = basic.overlap if self._basic else overlap
        return operation(
            self.df,
            accessor=self,
        )

    def non_overlap(self) -> pd.DataFrame:
        """Return the intervals in a DataFrame which overlap with each other.

        Example:
        ```
        #     A  :   [----)     [----)
        #               [----)
        #
        # Result :              [----)
        ```

            >>> df = pd.DataFrame([(100, 220), (200, 250), (400, 500)], columns=['start', 'end'])
            >>> df.ivl.non_overlap()
               start    end
            2  400.0  500.0

        Compatible with `basic` flag.

            >>> df.ivl.basic.non_overlap()
               start    end
            2  400.0  500.0

        """
        operation = basic.non_overlap if self._basic else non_overlap
        return operation(
            self.df,
            accessor=self,
        )

    # TODO Test unit duplicates
    def intersection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the intervals in a DataFrame which intersect with the intervals in another
        DataFrame.

        Example:
        ```
        #     A  :   [----)        [----)
        #               [----)
        #     B  :   [----) [----)        [----)
        #
        # Result :   [----) [----)
        #               [----)
        ```

            >>> df_a = pd.DataFrame([(100, 220), (200, 250), (400, 500)], columns=['start', 'end'])
            >>> df_b = pd.DataFrame([(100, 220), (230, 300), (600, 700)], columns=['start', 'end'])
            >>> df_a.ivl.intersection(df_b)
               start    end
            0  100.0  220.0
            1  200.0  250.0
            1  230.0  300.0

        Compatible with `basic` flag.

            >>> df_a.ivl.basic.intersection(df_b)
               start    end
            0  100.0  220.0
            1  200.0  250.0
            1  230.0  300.0

        Args:
            df: Another DataFrame of intervals.
        """
        operation = basic.intersection if self._basic else intersection
        return operation(
            self.df,
            self.format(df),
            accessor=self,
        )

    def diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the intervals in a DataFrame which do not intersect with the intervals in another
        DataFrame.

        Example:
        ```
        #     A  :   [----)        [----)
        #               [----)
        #     B  :   [----) [----)        [----)
        #
        # Result :                 [----)
        ```

            >>> df_a = pd.DataFrame([(100, 220), (200, 250), (400, 500)], columns=['start', 'end'])
            >>> df_b = pd.DataFrame([(100, 220), (230, 300), (600, 700)], columns=['start', 'end'])
            >>> df_a.ivl.diff(df_b)
               start    end
            2  400.0  500.0

        Compatible with `basic` flag.

            >>> df_a.ivl.basic.diff(df_b)
               start    end
            2  400.0  500.0

        Args:
            df: Another DataFrame of intervals.
        """
        operation = basic.diff if self._basic else diff
        return operation(
            self.df,
            self.format(df),
            accessor=self,
        )

    def symdiff(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the intervals in a DataFrame which are mutually exclusive with intervals in
        another DataFrame.

        Example:
        ```
        #     A  :   [----)        [----)
        #               [----)
        #     B  :   [----) [----)        [----)
        #
        # Result :                 [----) [----)
        ```

            >>> df_a = pd.DataFrame([(100, 220), (200, 250), (400, 500)], columns=['start', 'end'])
            >>> df_b = pd.DataFrame([(100, 220), (230, 300), (600, 700)], columns=['start', 'end'])
            >>> df_a.ivl.symdiff(df_b)
               start    end
            2  400.0  500.0
            2  600.0  700.0

        Compatible with `basic` flag.
            >>> df_a.ivl.symdiff(df_b)
               start    end
            2  400.0  500.0
            2  600.0  700.0

        Args:
            df: Another DataFrame of intervals.
        """
        operation = basic.symdiff if self._basic else symdiff
        return operation(
            self.df,
            self.format(df),
            accessor=self,
        )

    def combine(self, *dfs) -> pd.DataFrame:
        """Return the combination of intervals in a DataFrame (optionally combine intervals across
        other DataFrames).

        Example:
        ```
        #     A  :   [----)        [----)
        #               [----)
        #
        # Result :   [-------)     [----)
        ```

            >>> df_a = pd.DataFrame([(100, 220), (200, 250), (400, 500)], columns=['start', 'end'])
            >>> df_a.ivl.combine()
               start    end
            0  100.0  250.0
            1  400.0  500.0
            >>> df_b = pd.DataFrame([(100, 220), (230, 300), (600, 700)], columns=['start', 'end'])
            >>> df_a.ivl.combine(df_b)
               start    end
            0  100.0  300.0
            1  400.0  500.0
            2  600.0  700.0

        Compatible with `basic` flag.

            >>> df_a.ivl.basic.combine(df_b)
               start    end
            0  100.0  300.0
            1  400.0  500.0
            2  600.0  700.0

        Args:
            dfs: Other DataFrames of intervals that are optionally unioned before combining
                intervals.
        """
        operation = basic.combine if self._basic else combine
        return operation(
            self.union(*dfs),
            accessor=self,
        ).reset_index(drop=True)

    def complement(
        self,
        left_bound: Optional[float] = None,
        right_bound: Optional[float] = None,
    ):
        """Return the complement of a DataFrame of intervals.

        Example:
        ```
        #     A  :   [----)        [----)    [---)
        #               [----)
        #
        # Result :           [-----)    [----)
        ```
            >>> df = pd.DataFrame(
            ...     [(100, 160), (200, 250), (400, 500), (450, 520)],
            ...     columns=['start', 'end'],
            ... )
            >>> df.ivl.complement()
               start    end
            0  160.0  200.0
            1  250.0  400.0

        Compatible with `basic` flag.

            >>> df.ivl.basic.complement()
               start    end
            0  160.0  200.0
            1  250.0  400.0

        """
        operation = basic.complement if self._basic else complement
        return operation(
            self.df,
            accessor=self,
            left_bound=left_bound,
            right_bound=right_bound,
        ).reset_index(drop=True)

    def truncate(self, df: pd.DataFrame):
        """Return the truncation of intervals in a DataFrame around the intervals of another
        DataFrame.

        Example:

        ```
        #     A  :   [------------)   [----)     [---)
        #     B  :      [----)            [----)        [---)
        #
        # Result :   [--)    [----)   [---)      [---)
        ```

            >>> df_a = pd.DataFrame([(100, 160), (200, 300), (400, 500)], columns=['start', 'end'])
            >>> df_b = pd.DataFrame([(140, 180), (230, 250), (600, 700)], columns=['start', 'end'])
            >>> df_a.ivl.truncate(df_b)
               start    end
            0  100.0  140.0
            1  200.0  230.0
            1  250.0  300.0
            2  400.0  500.0

        Compatible with `basic` flag.

            >>> df_a.ivl.basic.truncate(df_b)
               start    end
            0  100.0  140.0
            1  200.0  230.0
            1  250.0  300.0
            2  400.0  500.0

        Args:
            df: Another DataFrame of intervals.
        """
        operation = basic.truncate if self._basic else truncate
        return operation(
            self.df,
            self.format(df),
            accessor=self,
        )

    def nearest(self, df: pd.DataFrame) -> pd.Series:
        """Return the distance between each interval and the closest interval in another DataFrame
        of intervals.

            >>> df_a = pd.DataFrame([(100, 160), (300, 370), (400, 500)], columns=['start', 'end'])
            >>> df_b = pd.DataFrame([(140, 180), (230, 250), (600, 700)], columns=['start', 'end'])
            >>> df_a.ivl.nearest(df_b)
            0      0.0
            1     50.0
            2    100.0
            dtype: float64

        Compatible with `basic` flag.

            >>> df_a.ivl.nearest(df_b)
            0      0.0
            1     50.0
            2    100.0
            dtype: float64

        Args:
            df: Another DataFrame of intervals.
        """
        operation = basic.nearest if self._basic else nearest
        return operation(
            self.df,
            self.format(df),
            accessor=self,
        )


def setup_ivl_accessor():
    register = pd.api.extensions.register_dataframe_accessor("ivl")
    register(IntervalsAccessor)
