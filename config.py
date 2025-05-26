import os
from enum import IntEnum as _IntEnum

import numpy as np
from screeninfo import Monitor as _Monitor
import peyes

from constants import *

EXPERIMENT_NAME = "v4"      # chane to v5 when analyzing newer subjects
STIMULI_VERSION = 1

## PATHS ##
_BASE_PATH = r"S:\Lab-Shared\Experiments\LWS Free Viewing Demo"
RAW_DATA_PATH = os.path.join(_BASE_PATH, "RawData")
OUTPUT_PATH = os.path.join(_BASE_PATH, "Results")

IMAGE_DIR_PATH = r"S:\Lab-Shared\Experiments\N170 free scan\ClutteredObjects_scan\Origional_Objects_Pics\organized"
SEARCH_ARRAY_PATH = os.path.join(_BASE_PATH, "Stimuli")

## Screen Monitor ##
TOBII_MONITOR = _Monitor(
    width=1920, height=1080,
    width_mm=530, height_mm=300,
    x=0, y=0, name="tobii", is_primary=True,
)
TOBII_PIXEL_SIZE_MM = np.mean([
    TOBII_MONITOR.width_mm / TOBII_MONITOR.width,
    TOBII_MONITOR.height_mm / TOBII_MONITOR.height,
])
TOBII_MISSING_VALUES = [-1, "-1", "-1.#IND0", np.nan, MISSING_VALUE]


## Eye Movement Detection ##
DETECTION_ALGORITHM = "Engbert"
MIN_EVENT_DURATION_MS = 5
PAD_BLINKS_MS = 0
DETECTOR = peyes.create_detector(DETECTION_ALGORITHM, MISSING_VALUE, MIN_EVENT_DURATION_MS, PAD_BLINKS_MS)


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
