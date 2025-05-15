import os
import io
from typing import Union, Optional, List
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cnfg
from data_models.LWSEnums import SexEnum, DominantHandEnum, DominantEyeEnum
from data_models.Trial import Trial


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


class Subject:
    """
    This class represents a single subject-session pair.
    If a subject performed more than one session, they will be represented as two separate objects.

    Each subject is defined by their personal information, and the series of trials they performed during the session.
    """

    __E_PRIME_FIELDS = {
        "Name": "name", "Subject": "subject_id", "Age": "age", "Sex": "sex", "Handedness": "hand", "DominantEye": "eye",
        "Session": "session", "SessionDate": "session_date", "SessionTime": "session_time", "Distance": "screen_distance_cm",
    }

    def __init__(
            self,
            exp_name: str,
            subject_id: int,
            name: str,
            age: float,
            sex: Union[str, SexEnum],
            hand: Union[str, DominantHandEnum],
            eye: Union[str, DominantEyeEnum],
            screen_distance_cm: float,
            session: int,
            date_time: Optional[Union[str, datetime]] = None,
            trials: Optional[List[Trial]] = None,
    ):
        self._experiment_name = exp_name
        self._id = subject_id
        self._name = name
        self._age = age
        self._sex = sex if isinstance(sex, SexEnum) else SexEnum(sex)
        self._hand = hand if isinstance(hand, DominantHandEnum) else DominantHandEnum(hand)
        self._eye = eye if isinstance(eye, DominantEyeEnum) else DominantEyeEnum(eye)
        self._screen_distance_cm = screen_distance_cm
        self._session = session
        try:
            self._date_time = date_time if isinstance(date_time, datetime) else datetime.strptime(date_time, cnfg.DATE_TIME_FORMAT)
        except (ValueError, TypeError):
            self._date_time = None
        trials = trials or []
        self._trials = sorted(trials, key=lambda trial: trial.trial_num)

    @staticmethod
    def from_raw(
            exp_name: str, subject_id: int, session: int, dirname: Optional[str] = None, verbose: bool = False,
    ) -> "Subject":
        """
        Reads the subject information from the raw data file and returns a Subject object.
        If a dirname is provided, uses this to find the file, otherwise constructs the dirname from the experiment name,
        subject ID, and session number.
        """
        if verbose:
            print("#####################################")
            print(f"Experiment: {exp_name}\tSubject: {subject_id}\tSession: {session}")
            print("Reading subject data...")
        file_prefix = f"{exp_name}-{subject_id}-{session}"
        dirname = dirname or file_prefix

        # read subject personal information
        subject_info = Subject.__parse_subject_info(os.path.join(cnfg.RAW_DATA_PATH, dirname, f"{file_prefix}.txt"))
        subject_info["exp_name"] = exp_name

        # read triggers and gaze data
        triggers, gaze = Subject.__parse_triggers_and_gaze(
            os.path.join(cnfg.RAW_DATA_PATH, dirname, f"{file_prefix}-TriggerLog.txt"),
            os.path.join(cnfg.RAW_DATA_PATH, dirname, f"{file_prefix}-GazeData.txt"),
        )

        # split the data into trials
        trials = []
        for trial_num in tqdm(
            triggers[cnfg.TRIAL_STR].dropna().unique().astype(int),
            desc="Trials", disable=not verbose,
        ):
            trial_triggers = triggers[triggers[cnfg.TRIAL_STR] == trial_num]
            trial_gaze = gaze[gaze[cnfg.TRIAL_STR] == trial_num]
            trial = Trial.from_frames(trial_triggers, trial_gaze)
            trials.append(trial)
        trials = sorted(trials, key=lambda t: t.trial_num)

        # create the subject object
        return Subject(trials=trials, **subject_info)

    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def age(self) -> float:
        return self._age

    @property
    def sex(self) -> SexEnum:
        return self._sex

    @property
    def hand(self) -> DominantHandEnum:
        return self._hand

    @property
    def eye(self) -> DominantEyeEnum:
        return self._eye

    @property
    def screen_distance_cm(self) -> float:
        return self._screen_distance_cm

    @property
    def session(self) -> int:
        return self._session

    @property
    def date_time(self) -> Optional[datetime]:
        return self._date_time

    @property
    def num_trials(self) -> int:
        return len(self._trials)

    def get_trials(self) -> List[Trial]:
        return self._trials

    def __repr__(self) -> str:
        return f"{self.experiment_name.upper()}-Subject {self.id}"

    @staticmethod
    def __parse_subject_info(file_path) -> dict:
        """ Reads subject's personal information from the E-Prime log file, and returns a dictionary with the information. """
        f = io.open(file_path, mode="r", encoding="utf-16")
        lines = f.readlines()
        subject_info = {field: None for field in cnfg.SUBJECT_INFO_FIELD_MAP.values()}
        for line in lines:
            if not ":" in line:
                continue
            eprime_field, value = line.split(":", 1)
            eprime_field, value = eprime_field.strip(), value.strip()
            if eprime_field in cnfg.SUBJECT_INFO_FIELD_MAP.keys():
                field = cnfg.SUBJECT_INFO_FIELD_MAP[eprime_field]
                subject_info[field] = value
        f.close()

        # Convert to numeric types
        subject_info["subject_id"] = int(subject_info["subject_id"])
        subject_info["age"] = float(subject_info["age"])
        subject_info["screen_distance_cm"] = float(subject_info["screen_distance_cm"])
        subject_info["session"] = int(subject_info["session"])

        # Convert the session date and time to a datetime object
        date = subject_info.pop("session_date", None)
        time = subject_info.pop("session_time", None)
        if date and time:
            subject_info["date_time"] = datetime.strptime(f"{date} {time}", cnfg.DATE_TIME_FORMAT)
        else:
            subject_info["date_time"] = None
        return subject_info

    @staticmethod
    def __parse_triggers_and_gaze(triggers_path, gaze_path) -> (pd.DataFrame, pd.DataFrame):
        """
        Parses gaze and triggers data:
        1. Reads the Tobii gaze and trigger log files
        2. Merges the two dataframes based on timestamp, to align the data
        3. Add columns to indicate block number, trial number, and whether data was recorded
        4. Splits the merged dataframe back into gaze and trigger dataframes and returns them
        """
        # read triggers
        triggers = pd.read_csv(triggers_path, sep="\t")
        triggers.rename(columns=cnfg.TRIGGER_FIELD_MAP, inplace=True)
        triggers[cnfg.TRIGGER_STR] = triggers[cnfg.TRIGGER_STR].map(lambda trgr: cnfg.ExperimentTriggerEnum(trgr))

        # read gaze data
        gaze = pd.read_csv(gaze_path, sep="\t")
        gaze.rename(columns=cnfg.TOBII_FIELD_MAP, inplace=True)
        # replace missing/invalid values to NaN
        et_cols = [
            cnfg.LEFT_X_STR, cnfg.LEFT_Y_STR, cnfg.RIGHT_X_STR, cnfg.RIGHT_Y_STR, cnfg.LEFT_PUPIL_STR, cnfg.RIGHT_PUPIL_STR
        ]
        gaze[et_cols] = gaze[et_cols].replace(cnfg.TOBII_MISSING_VALUES, cnfg.MISSING_VALUE, inplace=False)
        gaze.loc[gaze["GazePointValidityLeftEye"] == 0, [cnfg.LEFT_X_STR, cnfg.LEFT_Y_STR]] = cnfg.MISSING_VALUE
        gaze.loc[gaze["GazePointValidityRightEye"] == 0, [cnfg.RIGHT_X_STR, cnfg.RIGHT_Y_STR]] = cnfg.MISSING_VALUE
        gaze.loc[gaze["PupilValidityLeftEye"] == 0, cnfg.LEFT_PUPIL_STR] = cnfg.MISSING_VALUE
        gaze.loc[gaze["PupilValidityRightEye"] == 0, cnfg.RIGHT_PUPIL_STR] = cnfg.MISSING_VALUE
        gaze = gaze.astype({col: float for col in et_cols})

        # correct to tobii's resolution
        gaze[cnfg.LEFT_X_STR] *= cnfg.TOBII_MONITOR.width
        gaze[cnfg.LEFT_Y_STR] *= cnfg.TOBII_MONITOR.height
        gaze[cnfg.RIGHT_X_STR] *= cnfg.TOBII_MONITOR.width
        gaze[cnfg.RIGHT_Y_STR] *= cnfg.TOBII_MONITOR.height

        # merge triggers and gaze to align timing
        merged = pd.merge(gaze, triggers, how='outer', on=[cnfg.TIME_STR])  # merge on time

        # add block column
        merged[cnfg.BLOCK_STR] = np.nan
        for trg in cnfg.ExperimentTriggerEnum:
            name = trg.name
            if not name.startswith("BLOCK_"):
                continue
            block_num = int(name.split("_")[-1])
            is_block_start = merged[cnfg.TRIGGER_STR].eq(trg)
            if not is_block_start.any():
                # no such block trigger - skip
                continue
            start_idx = is_block_start.idxmax()  # find the first occurrence of the block trigger
            merged.loc[merged.index[start_idx:], cnfg.BLOCK_STR] = block_num

        # add trial column
        is_trial = _is_between_triggers(
            merged[cnfg.TRIGGER_STR], cnfg.ExperimentTriggerEnum.TRIAL_START, cnfg.ExperimentTriggerEnum.TRIAL_END
        )
        is_trial_start = is_trial.ne(is_trial.shift()) & is_trial  # find the start of each trial
        trial_num = is_trial_start.cumsum()  # assign trial numbers
        trial_num.loc[~is_trial] = np.nan  # set non-trial rows to NaN
        merged[cnfg.TRIAL_STR] = trial_num

        # add `is_recording` columns
        merged['is_recording'] = _is_between_triggers(
            merged[cnfg.TRIGGER_STR], cnfg.ExperimentTriggerEnum.START_RECORD, cnfg.ExperimentTriggerEnum.STOP_RECORD
        )

        # reorder columns
        cols_ord = [cnfg.TIME_STR, cnfg.TRIGGER_STR]
        cols_ord += [col for col in merged.columns if col not in cols_ord]
        merged = merged[cols_ord]

        # split back to gaze and triggers
        triggers = merged.loc[
            merged[cnfg.TRIGGER_STR].notna(), [cnfg.TIME_STR, cnfg.TRIGGER_STR, cnfg.TRIAL_STR, cnfg.BLOCK_STR, 'is_recording']]
        gaze_cols = [col for col in cnfg.TOBII_FIELD_MAP.values() if col != cnfg.TIME_STR]
        is_gaze = merged[gaze_cols].notna().any(axis=1)
        gaze = merged.loc[is_gaze, [cnfg.TIME_STR] + gaze_cols + [cnfg.TRIAL_STR, cnfg.BLOCK_STR, 'is_recording']]
        return triggers, gaze
