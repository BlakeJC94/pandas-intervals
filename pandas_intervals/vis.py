from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

Figure = go.Figure if go is not None else None


def plot_intervals(
    dfs: List[pd.DataFrame],
    colors: Optional[List[str]] = None,
    names: Optional[List[str]] = None,
    **layout_kwargs,
) -> Figure:
    if not colors:
        colors = ["red", "green", "blue"]

    if not names:
        names = [str(i) for i in range(len(dfs))]
    assert len(names) == len(dfs), "Expected number of names to match number of DataFrames."

    hspan = 1.0

    offset = 0.0
    traces = []
    tickvals, ticktext = [], []
    for i, df in enumerate(dfs):
        for interval in df.itertuples(index=False, name=None):
            traces.append(
                create_trace_from_interval(
                    interval,
                    offset=offset,
                    span=hspan,
                    color=colors[i % len(colors)],
                    name=names[i],
                )
            )
        tickvals.append(offset)
        ticktext.append(f"{i}   ")
        offset -= hspan

    y_pad = min((1 - 0.1 * hspan * len(dfs)) / 2, 0.1 * hspan)
    y_range = [offset - y_pad, hspan + y_pad]

    x_min = min(df["start"].min() for df in dfs)
    x_max = max(df["end"].max() for df in dfs)
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
