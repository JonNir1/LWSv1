import os
from typing import Union, Optional, List
from datetime import datetime

from tqdm import tqdm

import config as cnfg
import parse_data as prs
from data_models.LWSEnums import SexEnum, DominantHandEnum, DominantEyeEnum
from data_models.Trial import Trial


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
        subject_info = prs.parse_subject_info(os.path.join(cnfg.RAW_DATA_PATH, dirname, f"{file_prefix}.txt"))
        subject_info["exp_name"] = exp_name

        # read triggers and gaze data
        triggers_path = os.path.join(cnfg.RAW_DATA_PATH, dirname, f"{file_prefix}-TriggerLog.txt")
        gaze_path = os.path.join(cnfg.RAW_DATA_PATH, dirname, f"{file_prefix}-GazeData.txt")
        behavioral_data = prs.parse_behavioral_data(triggers_path, gaze_path)

        # correct to tobii's resolution
        behavioral_data[cnfg.LEFT_X_STR] *= cnfg.TOBII_MONITOR.width
        behavioral_data[cnfg.LEFT_Y_STR] *= cnfg.TOBII_MONITOR.height
        behavioral_data[cnfg.RIGHT_X_STR] *= cnfg.TOBII_MONITOR.width
        behavioral_data[cnfg.RIGHT_Y_STR] *= cnfg.TOBII_MONITOR.height

        # split the data into trials
        trials = []
        for trial_num in tqdm(
            behavioral_data[cnfg.TRIAL_STR].dropna().unique().astype(int),
            desc="Trials", disable=not verbose,
        ):
            trial_data = behavioral_data[behavioral_data[cnfg.TRIAL_STR] == trial_num]
            trial_data = trial_data.iloc[2:]  # first 2 rows are too early    # TODO: find a way to remove these by time-diff
            trial = Trial.from_behavior(trial_data)
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
