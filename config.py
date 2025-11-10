import os
from typing import Union, Literal

import plotly.express.colors as _colors

from constants import *
from data_models.LWSEnums import SubjectActionCategoryEnum

STIMULI_VERSION = 1

## PATHS ##
_BASE_PATH = r"S:\Lab-Shared\Experiments\LWS\Tobii Demo"
RAW_DATA_PATH = os.path.join(_BASE_PATH, "RawData")
OUTPUT_PATH = os.path.join(_BASE_PATH, "Results")
SUBJECT_OUTPUT_PATH = os.path.join(OUTPUT_PATH, f"{SUBJECT_STR}s")

IMAGE_DIR_PATH = r"S:\Lab-Shared\Experiments\N170 free scan\ClutteredObjects_scan\Origional_Objects_Pics\organized"
SEARCH_ARRAY_PATH = os.path.join(_BASE_PATH, "Stimuli")


## Analysis Parameters ##
### Pre-Processing Pipeline Parameters ###
IDENTIFICATION_ACTIONS = [     # list of subject-actions indicating target identification
    SubjectActionCategoryEnum.MARK_AND_CONFIRM,
    # SubjectActionCategoryEnum.MARK_ONLY    # uncomment this to include marking-only actions
]
BAD_ACTIONS = [
    act for act in SubjectActionCategoryEnum if
    act != SubjectActionCategoryEnum.NO_ACTION and act not in IDENTIFICATION_ACTIONS
]

ON_TARGET_THRESHOLD_DVA = 1.75          # threshold to determine if a gaze/fixation is on-target
MAX_GAZE_TO_TRIGGER_TIME_DIFF = 5       # Maximum allowed time difference between gaze and trigger events for them to be considered as part of the same event.
VISIT_MERGING_TIME_THRESHOLD = 1000     # temporal window for considering two events as part of the same visit, in milliseconds

### Funnel Analysis Parameters ###
GAZE_COVERAGE_PERCENT_THRESHOLD = 80    # minimum percent of trial time that gaze data must cover to be included in analysis
TIME_TO_TRIAL_END_THRESHOLD = 1000      # fixations/visits ending within this time from the trial end are considered not-LWS.
FIXATIONS_TO_STRIP_THRESHOLD = 3        # fixations/visits whose following number of fixations fall in the bottom strip are not considered LWS.

_ANY_FUNNEL_STEPS = [
    # sequence of steps to determine if a fixation/visit is valid (valid trial, valid fixation) and on-target
    "all",
    "trial_gaze_coverage",
    # "trial_has_actions",    # uncomment to exclude trials with no subject-actions
    "trial_no_bad_action",
    "trial_no_false_alarm",
    "instance_not_outlier",
    "instance_on_target",
]
LWS_FUNNEL_STEPS = _ANY_FUNNEL_STEPS + [
    # additional steps to determine if a valid & on-target fixation/visit is a Looking-without-Seeing (LWS) instance
    "instance_before_identification",
    "instance_not_close_to_trial_end",
    "not_before_exemplar_visit",  # fixations/visits that precede exemplar section (bottom-strip) visits are not LWS
    "final"
]
TARGET_RETURN_FUNNEL_STEPS = _ANY_FUNNEL_STEPS + [
    # additional steps to determine if a valid & on-target fixation/visit is a target-return instance
    "instance_after_identification",
    "final"
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
