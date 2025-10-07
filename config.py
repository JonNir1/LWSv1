import os
from typing import Union, Literal

import plotly.express.colors as _colors

from constants import *

STIMULI_VERSION = 1

## PATHS ##
_BASE_PATH = r"S:\Lab-Shared\Experiments\LWS\Tobii Demo"
RAW_DATA_PATH = os.path.join(_BASE_PATH, "RawData")
OUTPUT_PATH = os.path.join(_BASE_PATH, "Results")
SUBJECT_OUTPUT_PATH = os.path.join(OUTPUT_PATH, f"{SUBJECT_STR}s")

IMAGE_DIR_PATH = r"S:\Lab-Shared\Experiments\N170 free scan\ClutteredObjects_scan\Origional_Objects_Pics\organized"
SEARCH_ARRAY_PATH = os.path.join(_BASE_PATH, "Stimuli")


## Analysis Parameters ##
MAX_GAZE_TO_TRIGGER_TIME_DIFF = 5       # Maximum allowed time difference between gaze and trigger events for them to be considered as part of the same event.
ON_TARGET_THRESHOLD_DVA = 1.0           # threshold to determine if a gaze point / fixation is on a target, in degrees of visual angle
VISIT_MERGING_TIME_THRESHOLD = 1000     # temporal window for considering two events as part of the same chunk, in milliseconds
TIME_TO_TRIAL_END_THRESHOLD = 1000      # fixations/visits ending within this time from the trial end are considered not-LWS.
FIXATIONS_TO_STRIP_THRESHOLD = 3        # fixations/visits whose following number of fixations fall in the bottom strip are not considered LWS.

LWS_FUNNEL_STEPS = [
    # sequence of steps to determine if a fixation/visit is a Looking-without-Seeing (LWS) instance
    "all", "valid_trial", "not_outlier", "on_target", "before_identification", "fixs_to_strip", "not_end_with_trial", "is_lws"
]
TARGET_RETURN_FUNNEL_STEPS = [
    # sequence of steps to determine if a fixation/visit is a post-identification target-return instance
    "all", "valid_trial", "not_outlier", "on_target", "after_identification", "is_return"
]


## VISUALIZATION CONFIGURATION ##
_DISCRETE_COLORMAP = _colors.qualitative.Dark24
_CONTINUOUS_COLORMAP = _colors.sequential.Viridis
_GENERIC_COLOR = "#808080"  # gray color for generic cases

FONT_FAMILY, FONT_COLOR = "Calibri", "black"
TITLE_FONT = dict(family=FONT_FAMILY, size=26, color=FONT_COLOR)
SUBTITLE_FONT = dict(family=FONT_FAMILY, size=22, color=FONT_COLOR)
COMMENT_FONT = dict(family=FONT_FAMILY, size=14, color=FONT_COLOR)
AXIS_LABEL_FONT = dict(family=FONT_FAMILY, size=20, color=FONT_COLOR)
AXIS_TICK_FONT = dict(family=FONT_FAMILY, size=16, color=FONT_COLOR)
AXIS_LABEL_STANDOFF = 2

GRID_LINE_COLOR, GRID_LINE_WIDTH = "lightgray", 1
ZERO_LINE_WIDTH = 2 * GRID_LINE_WIDTH


def get_discrete_color(value: Union[Literal["all"], int], loop: bool = False) -> str:
    """
    Get a discrete color for a given value, either a specific integer or the string "all".
    If `loop` is False, raises an error if the value is not in the expected range.
    """
    if isinstance(value, str) and value.lower() == ALL_STR:
        return _GENERIC_COLOR
    try:
        new_value = float(value)
    except ValueError:
        raise TypeError(f"Value must be an integer or 'all', got `{value}` of type {type(value)}.")
    if new_value != int(new_value):
        raise TypeError(f"Value must be an integer or 'all', got `{value}` of type {type(value)}.")
    value = int(new_value)
    if loop:
        value = value % len(_DISCRETE_COLORMAP)
    assert 0 <= value < len(_DISCRETE_COLORMAP), f"Value {value} out of range for discrete colormap (0-{len(_DISCRETE_COLORMAP)-1})."
    return _DISCRETE_COLORMAP[value]
