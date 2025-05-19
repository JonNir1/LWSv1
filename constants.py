import numpy as np


MISSING_VALUE = np.nan
DATE_TIME_FORMAT = "%m-%d-%Y %H:%M:%S"

TIME_STR = "time"
TRIGGER_STR = "trigger"
ACTION_STR = "action"
LABEL_STR = "label"
EVENT_STR = "event"
TARGET_STR = "target"

X, Y = "x", "y"
PUPIL_STR = "pupil"
LEFT_STR, RIGHT_STR = "left", "right"
LEFT_X_STR, LEFT_Y_STR, LEFT_PUPIL_STR = f"{LEFT_STR}_{X}", f"{LEFT_STR}_{Y}", f"{LEFT_STR}_{PUPIL_STR}"
RIGHT_X_STR, RIGHT_Y_STR, RIGHT_PUPIL_STR = f"{RIGHT_STR}_{X}", f"{RIGHT_STR}_{Y}", f"{RIGHT_STR}_{PUPIL_STR}"
LEFT_LABEL_STR, RIGHT_LABEL_STR = f"{LEFT_STR}_{LABEL_STR}", f"{RIGHT_STR}_{LABEL_STR}"
LEFT_EVENT_STR, RIGHT_EVENT_STR = f"{LEFT_STR}_{EVENT_STR}", f"{RIGHT_STR}_{EVENT_STR}"

SUBJECT_STR, SESSION_STR = "subject", "session"
BLOCK_STR, TRIAL_STR = "block", "trial"
CONDITION_STR = "condition"
CATEGORY_STR = "category"
IMAGE_STR = "image"
IS_RECORDING_STR = "is_recording"

