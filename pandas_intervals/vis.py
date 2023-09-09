from typing import Dict, Any, List, Optional, Tuple, Union

import pandas as pd

from .utils import _df_groups

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

Figure = go.Figure if go is not None else None


def plot_intervals(
    df: pd.DataFrame,
    groupby_cols: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    **layout_kwargs,
) -> Figure:
    dfs = []
    for group, df in _df_groups([df], groupby_cols=groupby_cols):
        group_name = group if group is not None else ""
        dfs.append((str(group_name), df[0]))

    if title is None and groupby_cols is not None and len(groupby_cols) > 0:
        title = "Grouped by " + ", ".join([repr(c) for c in groupby_cols])

    return plot_interval_groups(dict(dfs), colors, title=title, **layout_kwargs)


def plot_interval_groups(
    dfs: Dict[str, pd.DataFrame],
    colors: Optional[List[str]] = None,
    **layout_kwargs,
) -> Figure:
    names, values = [], []
    for k, v in dfs.items():
        names.append(k)
        values.append(v)
    return _plot_interval_groups(*values, colors=colors, names=names, **layout_kwargs)


def _plot_interval_groups(
    *dfs,
    colors: Optional[List[str]] = None,
    names: Optional[List[str]] = None,
    **layout_kwargs,
) -> Figure:
    if not colors:
        colors = ["red", "green", "blue"]

    if not names:
        names = ["" for _ in range(len(dfs))]

    assert len(names) == len(dfs)
    dfs = [(name, df) for name, df in zip(names, dfs)]

    hspan = 1.0

    offset = 0.0
    traces = []
    tickvals, ticktext = [], []
    for i, (name, df) in enumerate(dfs):
        for interval in df.itertuples(index=False, name=None):
            traces.append(
                create_trace_from_interval(
                    interval,
                    offset=offset,
                    span=hspan,
                    color=colors[i % len(colors)],
                    name=name,
                )
            )
        tickvals.append(offset)
        ticktext.append(f"{name}   ")
        offset -= hspan

    y_pad = min((1 - 0.1 * hspan * len(dfs)) / 2, 0.1 * hspan)
    y_range = [offset - y_pad, hspan + y_pad]

    x_min = min(df["start"].min() for _, df in dfs)
    x_max = max(df["end"].max() for _, df in dfs)
    x_span = x_max - x_min
    x_pad = x_span * 0.1
    x_range = [x_min - x_pad, x_max + x_pad]

    fig = go.Figure(traces)
    fig.update_layout(showlegend=False, **layout_kwargs)
    fig.update_xaxes(
        range=x_range,
        gridcolor="lightgrey",
        linecolor="lightgrey",
        showspikes=True,
        spikecolor="black",
        spikesnap="cursor",
        spikemode="across",
        zeroline=False,
        showgrid=True,
    )
    fig.update_yaxes(
        range=y_range,
        tickvals=tickvals,
        ticktext=ticktext,
        zeroline=False,
        showgrid=False,
    )

    fig.show()
    return fig


def create_trace_from_interval(
    interval: Tuple[float],
    offset: float = 0.0,
    span: float = 1.0,
    color: str = "red",
    name: str = "",
) -> Dict[str, Any]:
    start, end, *metadata = interval
    lower, upper = offset - 0.4 * span, offset + 0.4 * span
    return go.Scatter(
        x=[start, start, end, end, start],
        y=[lower, upper, upper, lower, lower],
        mode="lines",
        name=name,
        showlegend=False,
        marker=dict(color=color),
        fill="toself",
        hoveron="fills",
        text=f"( {start} , {end} ] ({str(metadata)})",
    )
