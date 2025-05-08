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
    "RTTime": TIME_STR,
    "GazePointPositionDisplayXLeftEye": LEFT_X_STR, "GazePointPositionDisplayYLeftEye": LEFT_Y_STR, "PupilDiameterLeftEye": LEFT_PUPIL_STR,
    "GazePointPositionDisplayXRightEye": RIGHT_X_STR, "GazePointPositionDisplayYRightEye": RIGHT_Y_STR, "PupilDiameterRightEye": RIGHT_PUPIL_STR,
    "ImageNum": "image_num", "ConditionName": "condition",
    # "BlockNum": BLOCK_STR,                      # block number as recorded by Tobii - NOT USING THIS
    # "RunningSample": TRIAL_STR,                 # trial number as recorded by Tobii - NOT USING THIS
    # "TrialNum": f"{TRIAL_STR}_in_{BLOCK_STR}",  # trial-in-block number as recorded by Tobii - NOT USING THIS

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

    # add block column
    merged[BLOCK_STR] = np.nan
    for trg in ExperimentTriggerEnum:
        name = trg.name
        if not name.startswith("BLOCK_"):
            continue
        block_num = int(name.split("_")[-1])
        is_block_start = merged[TRIGGER_STR].eq(trg)
        if not is_block_start.any():
            # no such block trigger - skip
            continue
        start_idx = is_block_start.idxmax()  # find the first occurrence of the block trigger
        merged.loc[merged.index[start_idx:], BLOCK_STR] = block_num

    # add trial column
    is_trial = _is_between_triggers(
        merged[TRIGGER_STR], ExperimentTriggerEnum.TRIAL_START, ExperimentTriggerEnum.TRIAL_END
    )
    is_trial_start = is_trial.ne(is_trial.shift()) & is_trial  # find the start of each trial
    trial_num = is_trial_start.cumsum()     # assign trial numbers
    trial_num.loc[~is_trial] = np.nan       # set non-trial rows to NaN
    merged[TRIAL_STR] = trial_num

    # add `is_search_array` column
    is_search_array = _is_between_triggers(
        merged[TRIGGER_STR], ExperimentTriggerEnum.STIMULUS_ON, ExperimentTriggerEnum.STIMULUS_OFF
    )
    merged['is_search_array'] = is_search_array

    # add `is_recording` column
    merged['is_recording'] = _is_between_triggers(
        merged[TRIGGER_STR], ExperimentTriggerEnum.START_RECORD, ExperimentTriggerEnum.STOP_RECORD
    )

    # reorder columns
    cols_ord = [TIME_STR, TRIGGER_STR]
    cols_ord += [col for col in merged.columns if col not in cols_ord]
    merged = merged[cols_ord]
    return merged


def _read_triggers(file_path) -> pd.DataFrame:
    """ Reads the trigger log file and returns a dataframe with the trigger times. """
    triggers = pd.read_csv(file_path, sep="\t")
    triggers.rename(columns=__TRIGGER_FIELD_MAP, inplace=True)
    triggers[TRIGGER_STR] = triggers[TRIGGER_STR].map(lambda trg: ExperimentTriggerEnum(trg))

    # add trial column
    is_trial = _is_between_triggers(
        triggers[TRIGGER_STR], ExperimentTriggerEnum.TRIAL_START, ExperimentTriggerEnum.TRIAL_END
    )
    is_trial_start = is_trial.ne(is_trial.shift()) & is_trial   # find the start of each trial
    trial_num = is_trial_start.cumsum()                         # assign trial numbers
    trial_num.loc[~is_trial] = np.nan                           # set non-trial rows to NaN
    triggers[TRIAL_STR] = trial_num
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


def _is_between_triggers(triggers: pd.Series, start: int, end: int) -> pd.Series:
    """
    Returns a boolean series indicating whether the values in the 'triggers' series occur after 'start' and before 'end'.
    """
    start_idxs = np.nonzero(triggers == start)[0]
    end_idxs = np.nonzero(triggers == end)[0]
    start_end_idxs = np.vstack([start_idxs, end_idxs]).T
    res = pd.Series(np.full_like(triggers, False, dtype=bool))
    for (start, end) in start_end_idxs:
        res.iloc[start:end + 1] = True
    return res
