from typing import Dict, Any, List, Union, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None


def plot_intervals(
    intervals_group: Union[Dict[str, NDArray], List[NDArray], NDArray],
    names: Optional[str] = None,
    colors: Union[List[str], str] = "red",
    x_buffer: float = 200,
    y_buffer: float = 0.05,
) -> go.Figure:
    if not isinstance(intervals_group, list):
        intervals_group = [intervals_group]

    for i in range(len(intervals_group)):
        if isinstance(intervals_group[i], pd.DataFrame):
            intervals_group[i] = intervals_group[i][["start", "end"]].values

    if isinstance(colors, str):
        colors = [colors]
    n_groups = len(intervals_group)
    if len(colors) == 1:
        colors = n_groups * colors
    if len(colors) < n_groups:
        raise ValueError("Unexpected number of colors supplied.")

    if names is None:
        names = [f"Intervals {i+1}" for i in range(n_groups)]
    if len(names) < n_groups:
        raise ValueError()

    main_figure = make_subplots(
        rows=n_groups,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        subplot_titles=names,
    )

    for i, intervals_array in enumerate(intervals_group):
        for interval in intervals_array:
            trace = create_trace_from_interval(interval, y_buffer, colors[i])
            main_figure.append_trace(trace, row=i + 1, col=1)

    main_figure.update_yaxes(range=[0, 1], showticklabels=False)
    main_figure.update_xaxes(
        range=[
            np.min(np.concatenate(intervals_group, axis=0)) - x_buffer,
            np.max(np.concatenate(intervals_group, axis=0)) + x_buffer,
        ],
    )
    main_figure.update_layout(height=200 * len(intervals_group), xaxis_showgrid=False)
    return main_figure


def create_trace_from_interval(
    interval: NDArray,
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
