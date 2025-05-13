import os
from enum import IntEnum as _IntEnum
from screeninfo import Monitor as _Monitor

EXPERIMENT_NAME = "v4"      # chane to v5 when analyzing newer subjects

## PATHS ##
_BASE_PATH = r"S:\Lab-Shared\Experiments\LWS Free Viewing Demo"
RAW_DATA_PATH = os.path.join(_BASE_PATH, "RawData")

IMAGE_DIR_PATH = r"S:\Lab-Shared\Experiments\N170 free scan\ClutteredObjects_scan\Origional_Objects_Pics\organized"
SEARCH_ARRAY_PATH = r"S:\Lab-Shared\Experiments\LWS Free Viewing Demo\Stimuli"

## Screen Monitor ##
TOBII_MONITOR = _Monitor(
    width=1920, height=1080,
    width_mm=530, height_mm=300,
    x=0, y=0, name="tobii", is_primary=True,
)

## CONSTANTS ##
DATE_TIME_FORMAT = "%m-%d-%Y %H:%M:%S"

TIME_STR = "time"
TRIGGER_STR = "trigger"

X, Y = "x", "y"
PUPIL_STR = "pupil"
LEFT_STR, RIGHT_STR = "left", "right"
LEFT_X_STR, LEFT_Y_STR, LEFT_PUPIL_STR = f"{LEFT_STR}_{X}", f"{LEFT_STR}_{Y}", f"{LEFT_STR}_{PUPIL_STR}"
RIGHT_X_STR, RIGHT_Y_STR, RIGHT_PUPIL_STR = f"{RIGHT_STR}_{X}", f"{RIGHT_STR}_{Y}", f"{RIGHT_STR}_{PUPIL_STR}"

SUBJECT_STR, SESSION_STR = "subject", "session"
BLOCK_STR, TRIAL_STR = "block", "trial"
CONDITION_STR = "condition"
IMAGE_STR = "image"

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
