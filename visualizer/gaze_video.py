import math
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from constants import X, Y, TIME_STR, START_TIME_STR, END_TIME_STR
from data_models.LWSEnums import DominantEyeEnum
from visualizer._colors import (
    RGBA, DEFAULT_GAZE_DOT_COLOR, DEFAULT_GAZE_TRAIL_COLOR,
    DEFAULT_FIXATION_HIGHLIGHT_COLOR, DEFAULT_TARGET_COLORS,
)
from visualizer._filter import filter_to_trial
from visualizer._stimulus import _load_stimulus_pil, _draw_target_markings_pil


def create_gaze_video(
        trial: "Trial",
        gaze: pd.DataFrame,
        identifications: Optional[pd.DataFrame] = None,
        fixation_events: Optional[pd.DataFrame] = None,
        output_path: str = "gaze_video.mp4",
        fps: int = 30,
        downsample_factor: Optional[int] = None,
        gaze_dot_radius: int = 8,
        gaze_dot_color: RGBA = DEFAULT_GAZE_DOT_COLOR,
        trail_color: RGBA = DEFAULT_GAZE_TRAIL_COLOR,
        trail_line_width: int = 1,
        show_fixation_highlight: bool = True,
        fixation_highlight_radius: int = 40,
        fixation_highlight_color: RGBA = DEFAULT_FIXATION_HIGHLIGHT_COLOR,
        target_colors: Optional[dict[str, RGBA]] = None,
        marker_line_width: int = 3,
) -> str:
    """
    Create an MP4 video showing gaze position over time on the trial's stimulus.

    Parameters
    ----------
    trial : Trial object
    gaze : DataFrame with columns time, x, y (dominant eye coordinates).
        Can be a full table (filtered internally) or pre-filtered.
    identifications : optional DataFrame for target color-coding
    fixation_events : optional DataFrame with start_time, end_time, x, y
        for fixation highlighting
    output_path : path for the output MP4 file
    fps : frames per second of output video
    downsample_factor : if None, auto-computed from sampling rate and fps
    """
    gaze = filter_to_trial(gaze, trial)
    if fixation_events is not None:
        fixation_events = filter_to_trial(fixation_events, trial)

    x_col, y_col = _dominant_eye_columns(trial)
    gaze = gaze.dropna(subset=[x_col, y_col])
    if gaze.empty:
        raise ValueError("No valid gaze samples after filtering.")

    if downsample_factor is None:
        downsample_factor = _auto_downsample_factor(gaze, fps)
    sampled = gaze.iloc[::downsample_factor].reset_index(drop=True)

    base_img = _load_stimulus_pil(trial)
    base_img = _draw_target_markings_pil(
        base_img, trial, identifications,
        target_colors=target_colors,
        marker_line_width=marker_line_width,
    )

    width, height = base_img.size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trail_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    trail_draw = ImageDraw.Draw(trail_img)

    xs = sampled[x_col].to_numpy()
    ys = sampled[y_col].to_numpy()
    times = sampled[TIME_STR].to_numpy() if TIME_STR in sampled.columns else None

    prev_x, prev_y = None, None
    for i in range(len(sampled)):
        cx, cy = float(xs[i]), float(ys[i])

        if prev_x is not None:
            trail_draw.line(
                [(prev_x, prev_y), (cx, cy)],
                fill=trail_color, width=trail_line_width,
            )

        frame = Image.alpha_composite(base_img, trail_img)
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        dot_draw = ImageDraw.Draw(overlay)

        if show_fixation_highlight and fixation_events is not None and times is not None:
            if _is_during_fixation(times[i], fixation_events):
                r = fixation_highlight_radius
                dot_draw.ellipse(
                    [cx - r, cy - r, cx + r, cy + r],
                    fill=fixation_highlight_color,
                )

        r = gaze_dot_radius
        dot_draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=gaze_dot_color,
        )

        frame = Image.alpha_composite(frame, overlay)
        frame_rgb = frame.convert("RGB")
        writer.write(cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR))

        prev_x, prev_y = cx, cy

    writer.release()
    return output_path


def _dominant_eye_columns(trial: "Trial") -> tuple[str, str]:
    eye = trial._subject.eye
    if eye == DominantEyeEnum.LEFT:
        return "left_x", "left_y"
    return "right_x", "right_y"


def _auto_downsample_factor(gaze: pd.DataFrame, fps: int) -> int:
    if TIME_STR not in gaze.columns or len(gaze) < 2:
        return 1
    dt = np.median(np.diff(gaze[TIME_STR].to_numpy()))
    if dt <= 0 or not np.isfinite(dt):
        return 1
    sampling_rate = 1000.0 / dt
    return max(1, math.ceil(sampling_rate / fps))


def _is_during_fixation(time_ms: float, fixation_events: pd.DataFrame) -> bool:
    mask = (fixation_events[START_TIME_STR] <= time_ms) & (time_ms <= fixation_events[END_TIME_STR])
    return mask.any()
