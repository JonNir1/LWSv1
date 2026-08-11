import os
from typing import Union, Literal

import plotly.express.colors as _colors

from constants import *
from data_models.LWSEnums import SubjectActionCategoryEnum

STIMULI_VERSION = 1

## PATHS ##
IMAGE_DIR_PATH = r"S:\Lab-Shared\Experiments\N170 free scan\ClutteredObjects_scan\Origional_Objects_Pics\organized"

_BASE_PATH = r"S:\Lab-Shared\Experiments\LWS\Tobii Demo"
_BASE_PATH = r"C:\Users\nirjo\Desktop\HCNL\LWS"                         # TODO: remove me!

RAW_DATA_PATH = os.path.join(_BASE_PATH, "RawData")
SEARCH_ARRAY_PATH = os.path.join(_BASE_PATH, "Stimuli")
OUTPUT_PATH = os.path.join(_BASE_PATH, "Results")
SUBJECT_OUTPUT_PATH = os.path.join(OUTPUT_PATH, f"{SUBJECT_STR}s")
PUBLICATIONS_PATH = os.path.join(_BASE_PATH, "Publications")


## Analysis Parameters ##
### Eye-Movement Detection Parameters ###
# Duration bounds (ms) for `peyes` event detection. Events outside these bounds are flagged as outliers and are
# dropped by `read_data(drop_outliers=True)`, so these sit directly on the dependent variable.
# NOTE: the values below reproduce what the pipeline used implicitly before they were made explicit - the min values
# were set in code, the max values were inherited from `peyes` defaults and never chosen for this paradigm.
# FIXATION_MAX_DURATION_MS in particular is an open question (see CODE_REVIEW.md T3): long dwells on a not-yet
# identified target are theoretically the strongest LWS candidates, and 2500 ms currently removes 6 of ~117k
# fixations, all of them on-target.
MIN_EVENT_DURATION_MS = 5               # shortest event the detector will emit
FIXATION_MIN_DURATION_MS = 50
# Measured 2026-08-06 over 116,947 fixations: the right tail decays smoothly and monotonically with no secondary
# mode (p99 = 770 ms, p99.9 = 1468 ms, max = 2797 ms; 500-750 ms n=1271 falling to 2750-3000 ms n=1). There is no
# empirical bump to cut at, so the peyes default is kept rather than replaced by an arbitrary cut.
# TODO(T3): check the visual-search literature for a principled upper bound on fixation duration and adopt it here.
FIXATION_MAX_DURATION_MS = 2500
SACCADE_MIN_DURATION_MS = MIN_EVENT_DURATION_MS
SACCADE_MAX_DURATION_MS = 200

### Pre-Processing Pipeline Parameters ###
ON_TARGET_THRESHOLD_DVA = 1.75          # threshold to determine if a gaze/fixation is on-target
IDENTIFICATION_ACTIONS = [     # list of subject-actions indicating target identification
    SubjectActionCategoryEnum.MARK_AND_CONFIRM,
    # SubjectActionCategoryEnum.MARK_ONLY    # uncomment this to include marking-only actions
]

# NOTE: funnel behaviour lives in `analysis/helpers/funnels/funnel_config.py`, not here. This file previously also
# carried GAZE_COVERAGE_PERCENT_THRESHOLD, TIME_TO_TRIAL_END_THRESHOLD, FIXATIONS_TO_STRIP_THRESHOLD, BAD_ACTIONS and
# the *_FUNNEL_STEPS lists, none of which the funnel code read - two sources of truth describing different pipelines.
# They now live in funnel_config.py as DEFAULT_* constants and the criteria lists.


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
