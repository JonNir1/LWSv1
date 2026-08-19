import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from constants import (
    X, Y, TIME_STR, START_TIME_STR, END_TIME_STR,
    ACTION_STR, TARGET_STR, SUBJECT_STR, TRIAL_STR,
    IDENTIFICATION_CATEGORY_STR,
)
from data_models.LWSEnums import DominantEyeEnum, SubjectActionCategoryEnum
from visualizer._colors import (
    RGBA, DEFAULT_GAZE_DOT_COLOR, DEFAULT_GAZE_TRAIL_COLOR,
    DEFAULT_FIXATION_HIGHLIGHT_COLOR, DEFAULT_TARGET_COLORS,
)
from visualizer._filter import filter_to_trial
from visualizer._stimulus import _load_stimulus_pil, _draw_target_markings_pil


@dataclass
class _ActionMarker:
    time_ms: float
    cx: float
    cy: float
    action: SubjectActionCategoryEnum
    radius: float


_REJECT_FADE_MS = 500.0


def create_gaze_video(
        trial: "Trial",
        gaze: pd.DataFrame,
        identifications: Optional[pd.DataFrame] = None,
        fixation_events: Optional[pd.DataFrame] = None,
        actions: Optional[pd.DataFrame] = None,
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
        action_circle_radius_px: Optional[float] = None,
        action_line_width: int = 3,
        title: Optional[str] = None,
) -> str:
    """
    Create an MP4 video showing gaze position over time on the trial's stimulus.

    Parameters
    ----------
    trial : Trial object
    gaze : DataFrame with columns time, x, y (dominant eye coordinates).
    identifications : optional DataFrame for target color-coding and action positions
    fixation_events : optional DataFrame with start_time, end_time, x, y
        for fixation highlighting
    actions : optional DataFrame with time and action columns. When provided,
        subject actions are shown as circles around the marked target:
        solid for confirmed, dashed for rejected/unconfirmed.
    output_path : path for the output MP4 file
    fps : frames per second of output video
    downsample_factor : if None, auto-computed from sampling rate and fps
    action_circle_radius_px : radius of action circles in pixels.
        Defaults to ON_TARGET_THRESHOLD_DVA converted to px.
    action_line_width : line width for action circles
    """
    gaze = filter_to_trial(gaze, trial)
    if fixation_events is not None:
        fixation_events = filter_to_trial(fixation_events, trial)
    if actions is not None:
        actions = filter_to_trial(actions, trial)
    if identifications is not None:
        identifications = filter_to_trial(identifications, trial)

    x_col, y_col = _dominant_eye_columns(trial)
    gaze = gaze.dropna(subset=[x_col, y_col])
    if gaze.empty:
        raise ValueError("No valid gaze samples after filtering.")

    if downsample_factor is None:
        downsample_factor = _auto_downsample_factor(gaze, fps)
    sampled = gaze.iloc[::downsample_factor].reset_index(drop=True)

    action_markers = _build_action_markers(
        actions, identifications, trial, action_circle_radius_px,
    )

    base_img = _load_stimulus_pil(trial)
    base_img = _draw_target_markings_pil(
        base_img, trial, identifications,
        target_colors=target_colors,
        marker_line_width=marker_line_width,
    )

    if title is None:
        title = f"Subject {trial._subject.id}, Trial {trial.trial_num}"
    _render_title(base_img, title)

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

        if times is not None and action_markers:
            _draw_action_overlays(dot_draw, times[i], action_markers, action_line_width)

        frame = Image.alpha_composite(frame, overlay)
        frame_rgb = frame.convert("RGB")
        writer.write(cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR))

        prev_x, prev_y = cx, cy

    writer.release()
    return output_path


def _render_title(image: Image.Image, title: str) -> None:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("calibri.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (image.width - tw) // 2
    draw.rectangle([tx - 6, 4, tx + tw + 6, th + 12], fill=(0, 0, 0, 160))
    draw.text((tx, 6), title, fill=(255, 255, 255, 255), font=font)


def _build_action_markers(
        actions: Optional[pd.DataFrame],
        identifications: Optional[pd.DataFrame],
        trial: "Trial",
        radius_px: Optional[float],
) -> list[_ActionMarker]:
    if actions is None or actions.empty:
        return []

    if radius_px is None:
        import pipeline.config as pcfg
        radius_px = pcfg.ON_TARGET_THRESHOLD_DVA / trial.px2deg

    targets = trial.get_targets()
    target_coords = {}
    for tgt_id, row in targets.iterrows():
        target_coords[tgt_id] = (row[f"{TARGET_STR}_x"], row[f"{TARGET_STR}_y"])

    markers = []
    for _, row in actions.iterrows():
        action_val = row[ACTION_STR]
        action_enum = SubjectActionCategoryEnum(int(action_val))
        if action_enum == SubjectActionCategoryEnum.ATTEMPTED_MARK:
            continue
        if action_enum == SubjectActionCategoryEnum.NO_ACTION:
            continue

        action_time = row[TIME_STR]
        cx, cy = _find_action_position(action_time, identifications, target_coords)
        if np.isnan(cx):
            continue

        markers.append(_ActionMarker(
            time_ms=action_time,
            cx=cx, cy=cy,
            action=action_enum,
            radius=radius_px,
        ))
    return markers


def _find_action_position(
        action_time: float,
        identifications: Optional[pd.DataFrame],
        target_coords: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    if identifications is None or identifications.empty:
        return np.nan, np.nan

    time_match = identifications.loc[
        np.isclose(identifications[TIME_STR], action_time, atol=1.0)
    ]
    if time_match.empty:
        return np.nan, np.nan

    row = time_match.iloc[0]
    tgt = row.get(TARGET_STR)
    if pd.notna(tgt) and tgt in target_coords:
        return target_coords[tgt]

    return np.nan, np.nan


def _draw_action_overlays(
        draw: ImageDraw.Draw,
        current_time: float,
        markers: list[_ActionMarker],
        line_width: int,
) -> None:
    for m in markers:
        if current_time < m.time_ms:
            continue

        if m.action == SubjectActionCategoryEnum.MARK_AND_CONFIRM:
            _draw_circle(draw, m.cx, m.cy, m.radius, line_width, solid=True)
        elif m.action == SubjectActionCategoryEnum.MARK_AND_REJECT:
            elapsed = current_time - m.time_ms
            if elapsed <= _REJECT_FADE_MS:
                alpha = int(255 * (1.0 - elapsed / _REJECT_FADE_MS))
                _draw_circle(draw, m.cx, m.cy, m.radius, line_width, solid=False, alpha=alpha)
        elif m.action == SubjectActionCategoryEnum.MARK_ONLY:
            _draw_circle(draw, m.cx, m.cy, m.radius, line_width, solid=False)


def _draw_circle(
        draw: ImageDraw.Draw,
        cx: float, cy: float, radius: float,
        line_width: int,
        solid: bool,
        alpha: int = 255,
) -> None:
    color = (0, 0, 0, alpha)
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    if solid:
        draw.ellipse(bbox, outline=color, width=line_width)
    else:
        _draw_dashed_circle(draw, cx, cy, radius, color, line_width, dash_count=24)


def _draw_dashed_circle(
        draw: ImageDraw.Draw,
        cx: float, cy: float, radius: float,
        color: tuple[int, ...],
        line_width: int,
        dash_count: int = 24,
) -> None:
    arc_len = 2 * math.pi / (dash_count * 2)
    for i in range(dash_count):
        start_angle = i * 2 * arc_len
        end_angle = start_angle + arc_len
        n_pts = max(3, int(arc_len * radius / 2))
        angles = np.linspace(start_angle, end_angle, n_pts)
        pts = [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]
        for j in range(len(pts) - 1):
            draw.line([pts[j], pts[j + 1]], fill=color, width=line_width)


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
