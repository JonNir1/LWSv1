from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter

from constants import X, Y, TOBII_MONITOR
from visualizer._colors import RGBA, DEFAULT_HEATMAP_COLORSCALE, DEFAULT_TARGET_COLORS
from visualizer._filter import filter_to_trial
from visualizer._stimulus import plot_stimulus_with_targets


def create_heatmap(
        trial: "Trial",
        gaze_data: pd.DataFrame,
        identifications: Optional[pd.DataFrame] = None,
        kernel_sigma: float = 30.0,
        heatmap_colorscale: str = DEFAULT_HEATMAP_COLORSCALE,
        heatmap_alpha: float = 0.5,
        show_colorbar: bool = True,
        title: Optional[str] = None,
        target_colors: Optional[dict[str, RGBA]] = None,
        marker_alpha: float = 0.4,
        marker_line_width: float = 2.5,
        output_path: Optional[str] = None,
) -> Figure:
    """
    Create a Gaussian-kernel heatmap overlaid on the trial's stimulus image.

    Parameters
    ----------
    trial : Trial object
    gaze_data : DataFrame with columns x, y, and optionally duration.
        If duration is present, it weights the Gaussian kernel.
        If absent, each point is weighted equally.
    identifications : optional DataFrame for target color-coding
    kernel_sigma : standard deviation of the Gaussian kernel in pixels
    heatmap_colorscale : matplotlib colormap name
    heatmap_alpha : opacity of the heatmap layer (0-1)
    show_colorbar : whether to add a colorbar
    output_path : if provided, saves the figure to this path
    """
    gaze_data = filter_to_trial(gaze_data, trial)

    width, height = TOBII_MONITOR.width, TOBII_MONITOR.height
    has_duration = "duration" in gaze_data.columns
    heatmap_array = _build_heatmap_array(
        gaze_data[X].to_numpy(),
        gaze_data[Y].to_numpy(),
        gaze_data["duration"].to_numpy() if has_duration else np.ones(len(gaze_data)),
        width, height, kernel_sigma,
    )

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    plot_stimulus_with_targets(
        ax, trial, identifications,
        target_colors=target_colors,
        marker_alpha=marker_alpha,
        marker_line_width=marker_line_width,
    )

    im = ax.imshow(
        heatmap_array,
        extent=[0, width, height, 0],
        cmap=heatmap_colorscale,
        alpha=heatmap_alpha,
        interpolation="bilinear",
    )
    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Density")

    if title is None:
        title = f"Subject {trial._subject.id}, Trial {trial.trial_num}"
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def _build_heatmap_array(
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        width: int,
        height: int,
        sigma: float,
) -> np.ndarray:
    heatmap = np.zeros((height, width), dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights)
    xi = np.clip(np.round(x[valid]).astype(int), 0, width - 1)
    yi = np.clip(np.round(y[valid]).astype(int), 0, height - 1)
    np.add.at(heatmap, (yi, xi), weights[valid])
    return gaussian_filter(heatmap, sigma=sigma)
