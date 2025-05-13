import os
from typing import Union, Optional
from datetime import datetime

import pandas as pd

import config as cnfg
import parse_data as prs
from data_models.LWSEnums import SexEnum, DominantHandEnum, DominantEyeEnum


class Subject:
    __E_PRIME_FIELDS = {
        "Name": "name", "Subject": "subject_id", "Age": "age", "Sex": "sex", "Handedness": "hand", "DominantEye": "eye",
        "Session": "session", "SessionDate": "session_date", "SessionTime": "session_time", "Distance": "screen_distance",
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
            screen_distance: float,
            session: int,
            date_time: Optional[Union[str, datetime]] = None,
            behavioral_data: Optional[pd.DataFrame] = None,
    ):
        self._experiment_name = exp_name
        self._id = subject_id
        self._name = name
        self._age = age
        self._sex = sex if isinstance(sex, SexEnum) else SexEnum(sex)
        self._hand = hand if isinstance(hand, DominantHandEnum) else DominantHandEnum(hand)
        self._eye = eye if isinstance(eye, DominantEyeEnum) else DominantEyeEnum(eye)
        self._screen_distance = screen_distance
        self._session = session
        try:
            self._date_time = date_time if isinstance(date_time, datetime) else datetime.strptime(date_time, cnfg.DATE_TIME_FORMAT)
        except (ValueError, TypeError):
            self._date_time = None
        self._behavioral_data = behavioral_data

    @staticmethod
    def from_raw(exp_name: str, subject_id: int, session: int, dirname: Optional[str] = None) -> "Subject":
        """
        Reads the subject information from the raw data file and returns a Subject object.
        If a dirname is provided, uses this to find the file, otherwise constructs the dirname from the experiment name,
        subject ID, and session number.
        """
        file_prefix = f"{exp_name}-{subject_id}-{session}"
        dirname = dirname or file_prefix

        # read subject personal information
        subject_info = prs.parse_subject_info(os.path.join(cnfg.RAW_DATA_PATH, dirname, f"{file_prefix}.txt"))
        subject_info["exp_name"] = exp_name

        # read triggers and gaze data
        triggers_path = os.path.join(cnfg.RAW_DATA_PATH, dirname, f"{file_prefix}-TriggerLog.txt")
        gaze_path = os.path.join(cnfg.RAW_DATA_PATH, dirname, f"{file_prefix}-GazeData.txt")
        behavioral_data = prs.parse_behavioral_data(triggers_path, gaze_path)

        # create the subject object
        return Subject(**subject_info, behavioral_data=behavioral_data)

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
    def screen_distance(self) -> float:
        return self._screen_distance

    @property
    def session(self) -> int:
        return self._session

    @property
    def date_time(self) -> Optional[datetime]:
        return self._date_time

    def get_behavior(self) -> Optional[pd.DataFrame]:
        return self._behavioral_data
