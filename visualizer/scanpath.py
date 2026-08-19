from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from constants import X, Y
from visualizer._colors import RGBA, DEFAULT_SCANPATH_COLORSCALE, DEFAULT_TARGET_COLORS
from visualizer._filter import filter_to_trial
from visualizer._stimulus import plot_stimulus_with_targets


def create_scanpath(
        trial: "Trial",
        fixations: pd.DataFrame,
        identifications: Optional[pd.DataFrame] = None,
        colorscale: str = DEFAULT_SCANPATH_COLORSCALE,
        min_dot_radius: int = 4,
        max_dot_radius: int = 20,
        dot_alpha: float = 0.8,
        line_width: float = 1.5,
        line_color: RGBA = (0, 0, 0, 180),
        show_legend: bool = True,
        target_colors: Optional[dict[str, RGBA]] = None,
        marker_alpha: float = 0.4,
        marker_line_width: float = 2.5,
        output_path: Optional[str] = None,
) -> Figure:
    """
    Draw a scanpath on the trial's stimulus image.

    Fixations are drawn as dots color-coded by temporal order and sized by duration.
    Consecutive fixations are connected by lines.

    Parameters
    ----------
    trial : Trial object
    fixations : DataFrame with columns x, y, duration, ordered temporally.
    identifications : optional DataFrame for target color-coding
    colorscale : matplotlib colormap name for temporal color coding
    min_dot_radius, max_dot_radius : range for duration-to-size mapping
    dot_alpha : alpha for fixation dots (0-1)
    line_width : saccade line width; 0 or NaN disables lines
    line_color : RGBA for saccade lines
    show_legend : whether to show a colorbar for fixation order
    output_path : if provided, saves the figure to this path
    """
    fixations = filter_to_trial(fixations, trial)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    plot_stimulus_with_targets(
        ax, trial, identifications,
        target_colors=target_colors,
        marker_alpha=marker_alpha,
        marker_line_width=marker_line_width,
    )

    xs = fixations[X].to_numpy()
    ys = fixations[Y].to_numpy()
    durations = fixations["duration"].to_numpy()
    n = len(fixations)
    if n == 0:
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    sizes = _duration_to_size(durations, min_dot_radius, max_dot_radius)
    t_norm = np.linspace(0, 1, n) if n > 1 else np.array([0.5])

    draw_lines = line_width is not None and np.isfinite(line_width) and line_width > 0
    if draw_lines:
        lc = tuple(c / 255 for c in line_color[:3]) + (line_color[3] / 255,)
        ax.plot(xs, ys, color=lc, linewidth=line_width, zorder=1)

    sc = ax.scatter(
        xs, ys,
        c=t_norm, cmap=colorscale,
        s=sizes, alpha=dot_alpha,
        edgecolors="black", linewidths=0.5,
        zorder=2,
    )

    if show_legend:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="Fixation order")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["First", "Last"])

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def _duration_to_size(
        durations: np.ndarray, min_size: int, max_size: int,
) -> np.ndarray:
    """Linear mapping from duration to matplotlib scatter point size (area)."""
    min_area = np.pi * min_size ** 2
    max_area = np.pi * max_size ** 2
    d_min, d_max = np.nanmin(durations), np.nanmax(durations)
    if d_max <= d_min:
        return np.full_like(durations, (min_area + max_area) / 2, dtype=float)
    t = (durations - d_min) / (d_max - d_min)
    return min_area + t * (max_area - min_area)
