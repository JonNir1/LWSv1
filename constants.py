import numpy as np
import pandas as pd
from screeninfo import Monitor as _Monitor

## GENERAL ##
MISSING_VALUE = np.nan
DATE_TIME_FORMAT = "%m-%d-%Y %H:%M:%S"
MILLISECONDS_IN_SECOND = 1000

## TOBII SCREEN TOBII_MONITOR ##
TOBII_MONITOR = _Monitor(
    width=1920, height=1080,
    width_mm=530, height_mm=300,
    x=0, y=0, name="tobii", is_primary=True,
)
TOBII_MISSING_VALUES = [-1, "-1", "-1.#IND0", np.nan, pd.NA, MISSING_VALUE]
PIXEL_SIZE_MM = np.mean([TOBII_MONITOR.width_mm / TOBII_MONITOR.width, TOBII_MONITOR.height_mm / TOBII_MONITOR.height,])

## STRINGS ##
SUBJECT_STR, SESSION_STR = "subject", "session"
BLOCK_STR, TRIAL_STR = "block", "trial"
TRIGGER_STR = "trigger"
ACTION_STR = "action"
LABEL_STR = "label"
EVENT_STR = "event"
CONDITION_STR = "condition"
CATEGORY_STR = "category"
TRIAL_CATEGORY_STR = f"{TRIAL_STR}_{CATEGORY_STR}"
DISTANCE_STR = "distance"
DISTANCE_PX_STR, DISTANCE_DVA_STR = f"{DISTANCE_STR}_px", f"{DISTANCE_STR}_dva"

TIME_STR = "time"
START_TIME_STR, END_TIME_STR = f"start_{TIME_STR}", f"end_{TIME_STR}"
X, Y = "x", "y"
PUPIL_STR = "pupil"
LEFT_STR, RIGHT_STR = "left", "right"
LEFT_X_STR, LEFT_Y_STR, LEFT_PUPIL_STR = f"{LEFT_STR}_{X}", f"{LEFT_STR}_{Y}", f"{LEFT_STR}_{PUPIL_STR}"
RIGHT_X_STR, RIGHT_Y_STR, RIGHT_PUPIL_STR = f"{RIGHT_STR}_{X}", f"{RIGHT_STR}_{Y}", f"{RIGHT_STR}_{PUPIL_STR}"
LEFT_LABEL_STR, RIGHT_LABEL_STR = f"{LEFT_STR}_{LABEL_STR}", f"{RIGHT_STR}_{LABEL_STR}"
LEFT_EVENT_STR, RIGHT_EVENT_STR = f"{LEFT_STR}_{EVENT_STR}", f"{RIGHT_STR}_{EVENT_STR}"

TARGET_STR = "target"
TARGET_DISTANCE_STR = f"{TARGET_STR}_{DISTANCE_STR}"
TARGET_TIME_STR = f"{TARGET_STR}_{TIME_STR}"
TARGET_CATEGORY_STR = f"{TARGET_STR}_{CATEGORY_STR}"

ALL_STR = "all"
IMAGE_STR = "image"
IS_RECORDING_STR = "is_recording"
METADATA_STR = "metadata"
FIXATION_STR = "fixation"
EYE_STR = "eye"
VISIT_STR = "visit"

IDENTIFIED_STR = "identified"
IDENTIFICATION_STR = "identification"
IDENTIFICATION_CATEGORY_STR = f"{IDENTIFICATION_STR}_{CATEGORY_STR}"

