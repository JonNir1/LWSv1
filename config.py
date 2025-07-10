import os
from enum import IntEnum as _IntEnum
from typing import Union, Literal

import peyes
import plotly.express.colors as _colors

from constants import *

STIMULI_VERSION = 1

## PATHS ##
_BASE_PATH = r"S:\Lab-Shared\Experiments\LWS Free Viewing Demo"
RAW_DATA_PATH = os.path.join(_BASE_PATH, "RawData")
OUTPUT_PATH = os.path.join(_BASE_PATH, "Results")
SUBJECT_OUTPUT_PATH = os.path.join(OUTPUT_PATH, f"{SUBJECT_STR}s")

IMAGE_DIR_PATH = r"S:\Lab-Shared\Experiments\N170 free scan\ClutteredObjects_scan\Origional_Objects_Pics\organized"
SEARCH_ARRAY_PATH = os.path.join(_BASE_PATH, "Stimuli")


## Eye Movement Detection ##
_DETECTION_ALGORITHM = "Engbert"
_MIN_EVENT_DURATION_MS = 5
_PAD_BLINKS_MS = 0
DETECTOR = peyes.create_detector(_DETECTION_ALGORITHM, MISSING_VALUE, _MIN_EVENT_DURATION_MS, _PAD_BLINKS_MS)


## Analysis Parameters ##
MAX_GAZE_TO_TRIGGER_TIME_DIFF = 20  # in ms    # Maximum allowed time difference between gaze and trigger events for them to be considered as part of the same event.
ON_TARGET_THRESHOLD_DVA = 1.0       # threshold to determine if a gaze point / fixation is on a target, in degrees of visual angle
VISIT_MERGING_TIME_THRESHOLD = 20   # temporal window for considering two events as part of the same chunk, in milliseconds
TIME_TO_TRIAL_END_THRESHOLD = 250   # fixations/visits ending within this time from the trial end are not considered LWS.
FIXATIONS_TO_STRIP_THRESHOLD = 3    # fixations/visits whose following number of fixations fall in the bottom strip are not considered LWS.


## Parsing Fields ##
SUBJECT_INFO_FIELD_MAP = {
    "Name": "name", "Subject": "subject_id", "Age": "age", "Sex": "sex", "Handedness": "hand", "DominantEye": "eye",
    "Session": "session", "SessionDate": "session_date", "SessionTime": "session_time", "Distance": "screen_distance_cm",
}
TRIGGER_FIELD_MAP = {"ClockTime": TIME_STR, "BioSemiCode": TRIGGER_STR}
TOBII_FIELD_MAP = {
    "RTTime": TIME_STR,
    "GazePointPositionDisplayXLeftEye": LEFT_X_STR,
    "GazePointPositionDisplayYLeftEye": LEFT_Y_STR,
    "PupilDiameterLeftEye": LEFT_PUPIL_STR,
    "GazePointPositionDisplayXRightEye": RIGHT_X_STR,
    "GazePointPositionDisplayYRightEye": RIGHT_Y_STR,
    "PupilDiameterRightEye": RIGHT_PUPIL_STR,
    "ImageNum": f"{IMAGE_STR}_num",
    "ConditionName": CONDITION_STR,
    # "BlockNum": BLOCK_STR,                      # block number as recorded by Tobii - NOT USING THIS
    # "RunningSample": TRIAL_STR,                 # trial number as recorded by Tobii - NOT USING THIS
    # "TrialNum": f"{TRIAL_STR}_in_{BLOCK_STR}",  # trial-in-block number as recorded by Tobii - NOT USING THIS

}


## Gaze and Trigger Columns ##
MUTUAL_COLUMNS = [TIME_STR, BLOCK_STR, TRIAL_STR, IS_RECORDING_STR]
TRIGGER_COLUMNS = [TRIGGER_STR, ACTION_STR]
GAZE_COLUMNS = [col for col in TOBII_FIELD_MAP.values() if col != TIME_STR]
DETECTOR_COLUMNS = [LEFT_LABEL_STR, RIGHT_LABEL_STR]


## VISUALIZATION CONFIGURATION ##
_DISCRETE_COLORMAP = _colors.qualitative.Dark24
_CONTINUOUS_COLORMAP = _colors.sequential.Viridis
_GENERIC_COLOR = "#808080"  # gray color for generic cases

FONT_FAMILY, FONT_COLOR = "Calibri", "black"
TITLE_FONT = dict(family=FONT_FAMILY, size=26, color=FONT_COLOR)
SUBTITLE_FONT = dict(family=FONT_FAMILY, size=20, color=FONT_COLOR)
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
    if isinstance(value, int):
        if loop:
            value = value % len(_DISCRETE_COLORMAP)
        assert 0 <= value < len(_DISCRETE_COLORMAP), f"Value {value} out of range for discrete colormap (0-{len(_DISCRETE_COLORMAP)-1})."
        return _DISCRETE_COLORMAP[value]
    if isinstance(value, float) and value == int(value):
        return get_discrete_color(int(value), loop=loop)
    raise TypeError(
        f"Value must be an integer or 'all', got `{value}` of type {type(value)}."
    )


## TRIGGERS ##
class ExperimentTriggerEnum(_IntEnum):
    """
    Experimental triggers for the experiment.
    Manually adapted from E-Prime's `.prm` files to define the triggers.
    """

    NULL = 0
    START_RECORD = 254
    STOP_RECORD = 255

    # Block Triggers
    BLOCK_1 = 101
    BLOCK_2 = 102
    BLOCK_3 = 103
    BLOCK_4 = 104
    BLOCK_5 = 105
    BLOCK_6 = 106
    BLOCK_7 = 107
    BLOCK_8 = 108
    BLOCK_9 = 109

    # Trial Triggers
    TRIAL_START = 11
    TRIAL_END = 12
    TARGETS_ON = 13             # targets screen
    TARGETS_OFF = 14
    STIMULUS_ON = 15            # search-array screen
    STIMULUS_OFF = 16

    # Key Presses
    SPACE_ACT = 211             # marks current gaze location as the target
    SPACE_NO_ACT = 212          # unable to mark current gaze location as the target
    CONFIRM_ACT = 221           # confirms choice of the target
    CONFIRM_NO_ACT = 222        # unable to confirm choice of the target
    NOT_CONFIRM_ACT = 231       # undo the choice of the target
    NOT_CONFIRM_NO_ACT = 232    # unable to undo the choice of the target
    OTHER_KEY = 241             # any other key pressed
    ABORT_TRIAL = 242           # user request to abort the trial
