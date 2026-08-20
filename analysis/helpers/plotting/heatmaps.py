from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


JET_COLORSCALE = [
    [0.0, "rgba(255, 255, 255, 0)"],
    [0.001, "rgba(255, 255, 255, 0)"],
    [0.01, "rgba(0, 0, 255, 1)"],
    [0.25, "rgba(0, 255, 255, 1)"],
    [0.5, "rgba(255, 255, 0, 1)"],
    [0.9, "rgba(255, 0, 0, 1)"],
]


def plot_screen_fixed_heatmaps(
        df: pd.DataFrame,
        group_col: str,
        mode: Literal["density", "probability"] = "density",
        title: str = None,
        screen_width: int = 1920,
        screen_height: int = 1080,
        colorscale: str | list = JET_COLORSCALE,
) -> go.Figure:
    """
    Generates heatmaps covering the full screen area (0 to screen_width/height).
    Empty areas will show the low-end of the colorscale (0) rather than gaps.
    """
    categories = sorted(df[group_col].unique().tolist())
    k = len(categories)

    cols = 2 if k in [2, 4] else 3
    sub_rows = int(np.ceil(k / cols))
    total_rows = sub_rows + 1

    fig = make_subplots(
        rows=total_rows, cols=cols,
        specs=[[{"colspan": cols}] + [None] * (cols - 1)] + [[{}] * cols for _ in range(sub_rows)],
        subplot_titles=["Overall"] + [str(cat).replace("_", " ") for cat in categories],
        row_heights=[0.5] + [(0.5 / sub_rows)] * sub_rows,
        vertical_spacing=0.05,
    )

    _add_fixed_grid_trace(
        fig, df, 1, 1,
        width=screen_width, height=screen_height,
        nbinsx=48, nbinsy=27,
        colorscale=colorscale,
        mode=mode,
    )

    for i, cat in enumerate(categories):
        subset = df[df[group_col] == cat]
        _add_fixed_grid_trace(
            fig, subset,
            row=(i // cols) + 2, col=(i % cols) + 1,
            width=screen_width, height=screen_height,
            nbinsx=48, nbinsy=27,
            colorscale=colorscale,
            mode=mode,
        )

    _apply_layout(fig, screen_width, screen_height, total_rows, title)
    return fig


def _add_fixed_grid_trace(
        fig: go.Figure,
        data: pd.DataFrame,
        row: int,
        col: int,
        nbinsx: int,
        nbinsy: int,
        width: int,
        height: int,
        colorscale,
        mode: str,
) -> None:
    x_edges = np.linspace(0, width, nbinsx + 1)
    y_edges = np.linspace(0, height, nbinsy + 1)

    z_total, _, _ = np.histogram2d(data["x"], data["y"], bins=[x_edges, y_edges])
    if mode.lower() == "density":
        z_final = z_total
    elif mode.lower() == "probability":
        success_subset = data[data["is_lws"] == 1]
        z_success, _, _ = np.histogram2d(success_subset["x"], success_subset["y"], bins=[x_edges, y_edges])
        with np.errstate(divide="ignore", invalid="ignore"):
            z_final = np.true_divide(z_success, z_total)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    z_final[z_total == 0] = np.nan

    trace = go.Heatmap(
        z=z_final.T,
        x=np.linspace(0, width, nbinsx),
        y=np.linspace(0, height, nbinsy),
        colorscale=colorscale,
        zmin=0, zmax=max([np.nanmax(z_final), 1]),
        zsmooth="best", connectgaps=False,
        hoverinfo="z+x+y", showscale=False,
    )
    fig.add_trace(trace, row=row, col=col)


def _apply_layout(
        fig: go.Figure,
        width: int,
        height: int,
        n_rows: int,
        title: str | None,
) -> None:
    fig_width = 1000
    aspect_ratio = height / width
    fig_height = fig_width * aspect_ratio * (n_rows * 0.85)

    fig.update_yaxes(
        range=[height, 0],
        showline=True, mirror=True, linecolor="black",
        showgrid=True,
        showticklabels=False,
    )
    fig.update_xaxes(
        range=[0, width],
        showline=True, mirror=True, linecolor="black",
        showgrid=True,
        showticklabels=False,
    )
    fig.update_layout(
        width=fig_width, height=int(fig_height),
        title=dict(text=title, x=0.5, xanchor="center"),
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40),
    )
