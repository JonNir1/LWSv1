from __future__ import annotations
import os
from typing import Union, Optional, List
from datetime import datetime

import pandas as pd
from tqdm import tqdm

import config as cnfg
from parse.subject_info import parse_subject_info
from parse.triggers_and_gaze import parse_triggers_and_gaze
from data_models.LWSEnums import SexEnum, DominantHandEnum, DominantEyeEnum


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
        self._trials: List["Trial"] = []

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
        subject_info = parse_subject_info(os.path.join(cnfg.RAW_DATA_PATH, dirname, f"{file_prefix}.txt"))
        subject_info["exp_name"] = exp_name
        subject = Subject(**subject_info)
        trials = subject.read_trials(dirname=dirname, verbose=verbose)
        for trial in trials:
            subject.add_trial(trial)
        if verbose:
            print(f"Subject {subject_id} has {len(subject._trials)} trials.")
            print("#####################################")
        return subject

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

    def get_trials(self, sort: bool = True) -> List["Trial"]:
        if not sort:
            return self._trials
        return sorted(self._trials, key=lambda t: t.trial_num)

    def add_trial(self, trial: "Trial") -> None:
        self._trials.append(trial)

    def read_trials(self, dirname: Optional[str] = None, verbose: bool = False) -> List["Trial"]:
        from data_models.Trial import Trial
        file_prefix = f"{self._experiment_name}-{self._id}-{self._session}"
        dirname = dirname or file_prefix
        if verbose:
            print(f"Reading trials from {dirname}.")
        triggers, gaze = parse_triggers_and_gaze(
            os.path.join(cnfg.RAW_DATA_PATH, dirname, f"{file_prefix}-TriggerLog.txt"),
            os.path.join(cnfg.RAW_DATA_PATH, dirname, f"{file_prefix}-GazeData.txt"),
        )
        trials = []
        for trial_num in tqdm(
                triggers[cnfg.TRIAL_STR].dropna().unique().astype(int),
                desc="Trials", disable=not verbose,
        ):
            # extract the trial data
            trial_triggers = triggers[triggers[cnfg.TRIAL_STR] == trial_num]
            trial_gaze = gaze[gaze[cnfg.TRIAL_STR] == trial_num]
            trial = Trial(self, trial_triggers.copy(), trial_gaze.copy())
            trials.append(trial)
        trials = sorted(trials, key=lambda t: t.trial_num)
        return trials

    def __repr__(self) -> str:
        return f"{self.experiment_name.upper()}-Subject {self.id}"
