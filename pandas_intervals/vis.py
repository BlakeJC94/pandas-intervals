from typing import Dict, Any, List, Union, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None

Figure = go.Figure if go is not None else None

# TODO Refactor
def plot_intervals(
    dfs: List[pd.DataFrame],
    names: Optional[str] = None,
    colors: Union[List[str], str] = "red",
    x_buffer: float = 200,
    y_buffer: float = 0.05,
) -> Figure:

    # for i in range(len(intervals_group)):
    #     if isinstance(intervals_group[i], pd.DataFrame):
    #         intervals_group[i] = intervals_group[i][["start", "end"]].values

    if isinstance(colors, str):
        colors = [colors]

    n_dfs = len(dfs)
    if len(colors) == 1:
        colors = n_dfs * colors
    if len(colors) < n_dfs:
        raise ValueError("Unexpected number of colors supplied.")

    if names is None:
        names = [f"Intervals {i+1}" for i in range(n_dfs)]
    if len(names) < n_dfs:
        raise ValueError()

    main_figure = make_subplots(
        rows=n_dfs,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        subplot_titles=names,
    )

    for i, df in enumerate(dfs):
        for j in range(len(df)):
            interval = df.iloc[j]['start'], df.iloc[j]['end']
            trace = create_trace_from_interval(interval, y_buffer, colors[i])
            main_figure.append_trace(trace, row=i + 1, col=1)

    main_figure.update_yaxes(range=[0, 1], showticklabels=False)
    main_figure.update_xaxes(
        range=[
            np.min(np.concatenate(dfs, axis=0)) - x_buffer,
            np.max(np.concatenate(dfs, axis=0)) + x_buffer,
        ],
    )
    main_figure.update_layout(height=200 * len(dfs), xaxis_showgrid=False)
    return main_figure


def create_trace_from_interval(
    interval: Tuple[float],
    y_buffer: float = 0,
    color: str = "red",
) -> Dict[str, Any]:
    start, end = interval
    lower, upper = y_buffer, 1 - y_buffer
    return go.Scatter(
        x=[start, start, end, end, start],
        y=[lower, upper, upper, lower, lower],
        mode="lines",
        name="",
        showlegend=False,
        marker={"color": color},
        fill="toself",
        hoveron="fills",
        text=f"[ {start} , {end} ]",
    )
