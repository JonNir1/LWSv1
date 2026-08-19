from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from PIL import Image, ImageDraw

import pipeline.config as pcfg
from constants import TARGET_STR
from visualizer._colors import RGBA, DEFAULT_TARGET_COLORS
from visualizer._filter import filter_to_trial


def _target_radius_px(trial: "Trial") -> float:
    return pcfg.ON_TARGET_THRESHOLD_DVA / trial.px2deg


def _resolve_target_color(
        target_id: str,
        identifications: Optional[pd.DataFrame],
        target_colors: dict[str, RGBA],
) -> RGBA:
    if identifications is None or identifications.empty:
        return target_colors.get("UNKNOWN", DEFAULT_TARGET_COLORS["UNKNOWN"])
    match = identifications.loc[identifications[TARGET_STR] == target_id]
    if match.empty:
        return target_colors.get("UNKNOWN", DEFAULT_TARGET_COLORS["UNKNOWN"])
    category = str(match.iloc[0]["identification_category"])
    return target_colors.get(category, target_colors.get("UNKNOWN", DEFAULT_TARGET_COLORS["UNKNOWN"]))


# -- matplotlib API (for static visualizations) --

def plot_stimulus_with_targets(
        ax: Axes,
        trial: "Trial",
        identifications: Optional[pd.DataFrame] = None,
        target_colors: Optional[dict[str, RGBA]] = None,
        marker_alpha: float = 0.4,
        marker_line_width: float = 2.5,
        title: Optional[str] = None,
) -> Axes:
    """
    Plot the trial's .bmp stimulus and draw circle markers at ON_TARGET_THRESHOLD_DVA
    radius around each target, color-coded by identification status.
    """
    colors = target_colors or DEFAULT_TARGET_COLORS
    if identifications is not None:
        identifications = filter_to_trial(identifications, trial)

    img = Image.open(trial._search_array.image_path)
    imshow_kwargs = dict(extent=[0, img.width, img.height, 0])
    if img.mode == "L":
        imshow_kwargs["cmap"] = "gray"
    ax.imshow(np.asarray(img), **imshow_kwargs)

    targets = trial.get_targets()
    radius_px = _target_radius_px(trial)

    for target_id, row in targets.iterrows():
        rgba = _resolve_target_color(target_id, identifications, colors)
        face_color = tuple(c / 255 for c in rgba[:3]) + (marker_alpha,)
        edge_color = tuple(c / 255 for c in rgba[:3]) + (1.0,)
        circle = mpatches.Circle(
            (row[f"{TARGET_STR}_x"], row[f"{TARGET_STR}_y"]),
            radius=radius_px,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=marker_line_width,
        )
        ax.add_patch(circle)

    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=14)
    return ax


# -- PIL API (for gaze_video.py) --

def _load_stimulus_pil(trial: "Trial") -> Image.Image:
    img = Image.open(trial._search_array.image_path)
    if img.mode == "L":
        img = img.convert("LA").convert("RGBA")
    else:
        img = img.convert("RGBA")
    return img


def _draw_target_markings_pil(
        image: Image.Image,
        trial: "Trial",
        identifications: Optional[pd.DataFrame] = None,
        target_colors: Optional[dict[str, RGBA]] = None,
        marker_line_width: int = 3,
) -> Image.Image:
    colors = target_colors or DEFAULT_TARGET_COLORS
    if identifications is not None:
        identifications = filter_to_trial(identifications, trial)

    result = image.copy()
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    targets = trial.get_targets()
    radius_px = _target_radius_px(trial)

    for target_id, row in targets.iterrows():
        rgba = _resolve_target_color(target_id, identifications, colors)
        cx, cy = row[f"{TARGET_STR}_x"], row[f"{TARGET_STR}_y"]
        bbox = [cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px]
        fill = (*rgba[:3], rgba[3] // 2)
        draw.ellipse(bbox, fill=fill, outline=rgba, width=marker_line_width)

    return Image.alpha_composite(result, overlay)
