import io
from datetime import datetime

import numpy as np
import pandas as pd

from config import *

__SUBJECT_INFO_FIELD_MAP = {
    "Name": "name", "Subject": "subject_id", "Age": "age", "Sex": "sex", "Handedness": "hand", "DominantEye": "eye",
    "Session": "session", "SessionDate": "session_date", "SessionTime": "session_time", "Distance": "screen_distance",
}
__TRIGGER_FIELD_MAP = {"ClockTime": TIME_STR, "BioSemiCode": TRIGGER_STR}
__TOBII_FIELD_MAP = {
    "RTTime": TIME_STR, "RunningSample": TRIAL_STR, "BlockNum": BLOCK_STR, "TrialNum": f"{TRIAL_STR}_in_{BLOCK_STR}",
    "GazePointPositionDisplayXLeftEye": LEFT_X_STR, "GazePointPositionDisplayYLeftEye": LEFT_Y_STR, "PupilDiameterLeftEye": LEFT_PUPIL_STR,
    "GazePointPositionDisplayXRightEye": RIGHT_X_STR, "GazePointPositionDisplayYRightEye": RIGHT_Y_STR, "PupilDiameterRightEye": RIGHT_PUPIL_STR,
    "ImageNum": "image_num", "ConditionName": "condition",
}
__TOBII_MISSING_VALUES = [-1, "-1", "-1.#IND0", np.nan]


def parse_subject_info(file_path) -> dict:
    """ Reads subject's personal information from the E-Prime log file, and returns a dictionary with the information. """
    f = io.open(file_path, mode="r", encoding="utf-16")
    lines = f.readlines()
    subject_info = {field: None for field in __SUBJECT_INFO_FIELD_MAP.values()}
    for line in lines:
        if not ":" in line:
            continue
        eprime_field, value = line.split(":", 1)
        eprime_field, value = eprime_field.strip(), value.strip()
        if eprime_field in __SUBJECT_INFO_FIELD_MAP.keys():
            field = __SUBJECT_INFO_FIELD_MAP[eprime_field]
            subject_info[field] = value
    f.close()

    # Convert to numeric types
    subject_info["subject_id"] = int(subject_info["subject_id"])
    subject_info["age"] = float(subject_info["age"])
    subject_info["screen_distance"] = float(subject_info["screen_distance"])
    subject_info["session"] = int(subject_info["session"])

    # Convert the session date and time to a datetime object
    date = subject_info.pop("session_date", None)
    time = subject_info.pop("session_time", None)
    if date and time:
        subject_info["date_time"] = datetime.strptime(f"{date} {time}", DATE_TIME_FORMAT)
    else:
        subject_info["date_time"] = None
    return subject_info


def parse_behavioral_data(triggers_path, gaze_path) -> pd.DataFrame:
    triggers = _read_triggers(triggers_path)
    gaze = _read_gaze(gaze_path)
    merged = pd.merge(gaze, triggers, how='outer', on=[TIME_STR])    # merge on time

    # get trial number from both sources
    trials_left = merged['trial_x'].fillna(merged['trial_y'])
    trials_right = merged['trial_y'].fillna(merged['trial_x'])
    assert trials_left.equals(trials_right), "Trials do not match between gaze and triggers data."
    assert trials_left.dropna().is_monotonic_increasing, "Trials are not in increasing order."
    merged[TRIAL_STR] = trials_left
    merged.drop(columns=['trial_x', 'trial_y'], inplace=True)   # remove duplicate trial columns

    # check if recording
    merged['is_recording'] = False
    rec_start_end_idxs = np.vstack([
        np.nonzero(merged[TRIGGER_STR] == ExperimentTriggerEnum.START_RECORD)[0],
        np.nonzero(merged[TRIGGER_STR] == ExperimentTriggerEnum.STOP_RECORD)[0]
    ]).T
    for (start, end) in rec_start_end_idxs:
        merged.loc[start:end + 1, 'is_recording'] = True

    # reorder columns
    cols_ord = [TIME_STR, TRIAL_STR, 'is_recording', TRIGGER_STR]
    cols_ord += [col for col in merged.columns if col not in cols_ord]
    merged = merged[cols_ord]
    return merged


def _read_triggers(file_path) -> pd.DataFrame:
    """ Reads the trigger log file and returns a dataframe with the trigger times. """
    triggers = pd.read_csv(file_path, sep="\t")
    triggers.rename(columns=__TRIGGER_FIELD_MAP, inplace=True)
    triggers[TRIGGER_STR] = triggers[TRIGGER_STR].map(lambda trg: ExperimentTriggerEnum(trg))

    # add trial column
    triggers[TRIAL_STR] = np.nan
    start_end_idxs = np.vstack([
        np.nonzero(triggers[TRIGGER_STR] == ExperimentTriggerEnum.TRIAL_START)[0],
        np.nonzero(triggers[TRIGGER_STR] == ExperimentTriggerEnum.TRIAL_END)[0]
    ]).T
    for i, (start, end) in enumerate(start_end_idxs):
        triggers.loc[start:end + 1, TRIAL_STR] = i + 1
    return triggers

def _read_gaze(file_path) -> pd.DataFrame:
    gaze = pd.read_csv(file_path, sep="\t")
    gaze.rename(columns=__TOBII_FIELD_MAP, inplace=True)

    # replace missing/invalid values to NaN
    et_cols = [LEFT_X_STR, LEFT_Y_STR, RIGHT_X_STR, RIGHT_Y_STR, LEFT_PUPIL_STR, RIGHT_PUPIL_STR]
    gaze[et_cols] = gaze[et_cols].replace(__TOBII_MISSING_VALUES, np.nan, inplace=False)
    gaze.loc[gaze["GazePointValidityLeftEye"] == 0, [LEFT_X_STR, LEFT_Y_STR]] = np.nan
    gaze.loc[gaze["GazePointValidityRightEye"] == 0, [RIGHT_X_STR, RIGHT_Y_STR]] = np.nan
    gaze.loc[gaze["PupilValidityLeftEye"] == 0, LEFT_PUPIL_STR] = np.nan
    gaze.loc[gaze["PupilValidityRightEye"] == 0, RIGHT_PUPIL_STR] = np.nan
    gaze = gaze.astype({col: float for col in et_cols})

    # return only the relevant columns
    return gaze[[col for col in gaze.columns if col in __TOBII_FIELD_MAP.values()]]

