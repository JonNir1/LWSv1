from __future__ import annotations
import os
import time
import pickle as pkl
from typing import Union, Optional, List
from datetime import datetime

import pandas as pd
from tqdm import tqdm

import config as cnfg
import helpers as hlp
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
        "Name": "name", "Age": "age", "Sex": "sex", "Handedness": "hand", "DominantEye": "eye",
        cnfg.SUBJECT_STR.capitalize(): f"{cnfg.SUBJECT_STR}_id",
        cnfg.SESSION_STR.capitalize(): cnfg.SESSION_STR,
        f"{cnfg.SESSION_STR.capitalize()}Date": f"{cnfg.SESSION_STR}_date",
        f"{cnfg.SESSION_STR.capitalize()}Time": f"{cnfg.SESSION_STR}_time",
        cnfg.DISTANCE_STR.capitalize(): "screen_distance_cm",
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
            exp_name: str, subject_id: int, session: int, data_dir: Optional[str] = None, verbose: bool = False,
    ) -> "Subject":
        """
        Reads the subject information from the raw data file and returns a Subject object.
        If a `data_dir` is provided, uses this to find the file, otherwise constructs the dirname from the experiment
        name, subject ID, and session number.
        """
        start = time.time()
        if verbose:
            print("#####################################")
            print(f"Experiment: {exp_name}\tSubject: {subject_id}\tSession: {session}")
            print("Reading subject data...")
        file_prefix = f"{exp_name}-{subject_id}-{session}"
        data_dir = data_dir or file_prefix
        subject_info = parse_subject_info(os.path.join(cnfg.RAW_DATA_PATH, data_dir, f"{file_prefix}.txt"))
        subject_info["exp_name"] = exp_name
        subject = Subject(**subject_info)
        trials = subject.read_trials(data_dir=data_dir, verbose=verbose)
        for trial in trials:
            subject.add_trial(trial)
        if verbose:
            print(f"Subject {subject_id} has {len(subject._trials)} trials.")
            print(f"Completed in {time.time() - start:.2f} seconds.")
            print("#####################################")
        return subject

    @staticmethod
    def from_pickle(
            path: Optional[str] = None, exp_name: Optional[str] = None, subject_id: Optional[int] = None
    ) -> "Subject":
        """
        Reads a Subject object from a pickle file, either from a specified `path` or by constructing the path using
        the experiment name and subject ID.
        :raise ValueError: If neither `path` nor both `exp_name` and `subject_id` are provided.
        :raise FileNotFoundError: If the file does not exist at the specified path.
        :return: The Subject object.
        """
        if not path and not (exp_name and subject_id):
            raise ValueError("Either `path` or both `exp_name` and `subject_id` must be provided to load a Subject from pickle.")
        path = path or Subject.get_pickle_path(exp_name, subject_id, makedirs=False)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")
        with open(path, "rb") as f:
            subject = pkl.load(f)
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

    @property
    def px2deg(self) -> float:
        """
        Returns the conversion factor from pixels to degrees of visual angle (DVA).
        To move from `d` pixels to DVA, use the formula: `d * self.px2deg`.
        """
        return hlp.convert_units(
            1, "px", "deg", cnfg.TOBII_PIXEL_SIZE_MM / 10, self._screen_distance_cm
        )

    @property
    def out_dir(self) -> str:
        """
        Returns the output directory for the subject's data, constructed as:
        <OUTPUT_PATH>/<experiment_name>/Subject_<subject_id>/
        If the directory does not exist, it will be created.
        """
        pickle_path = self.get_pickle_path(self._experiment_name, self._id, makedirs=True)
        return os.path.dirname(pickle_path)

    def get_trials(self, sort: bool = True) -> List["Trial"]:
        if not sort:
            return self._trials
        return sorted(self._trials, key=lambda t: t.trial_num)

    def add_trial(self, trial: "Trial") -> None:
        self._trials.append(trial)

    def read_trials(self, data_dir: Optional[str] = None, verbose: bool = False) -> List["Trial"]:
        from data_models.Trial import Trial
        file_prefix = f"{self._experiment_name}-{self._id}-{self._session}"
        data_dir = data_dir or file_prefix
        if verbose:
            print(f"Reading trials from {data_dir}.")
        triggers, gaze = parse_triggers_and_gaze(
            os.path.join(cnfg.RAW_DATA_PATH, data_dir, f"{file_prefix}-TriggerLog.txt"),
            os.path.join(cnfg.RAW_DATA_PATH, data_dir, f"{file_prefix}-GazeData.txt"),
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

    def to_pickle(self, overwrite: bool = False) -> str:
        """
        Saves the Subject object to a pickle file and returns the path to the file, constructed as:
        <OUTPUT_PATH>/<experiment_name>/Subject_<subject_id>/Subject.pkl
        If the file already exists and `overwrite` is False, it will raise a FileExistsError.
        """
        pickle_path = self.get_pickle_path(self._experiment_name, self._id, makedirs=True)
        if not os.path.exists(pickle_path):
            with open(pickle_path, "wb") as f:
                pkl.dump(self, f)
        elif overwrite:
            with open(pickle_path, "wb") as f:
                pkl.dump(self, f)
        else:
            raise FileExistsError(f"Pickle file already exists. Use `overwrite=True` to overwrite it.")
        return pickle_path

    @staticmethod
    def get_pickle_path(exp_name: str, subject_id: int, makedirs: bool) -> str:
        """
        Returns the path to the pickle file for the subject's data, constructed as:
        <OUTPUT_PATH>/<experiment_name>/Subject_<subject_id>/Subject.pkl
        If `makedirs` is True, the directory will be created if it does not exist.
        """
        out_dir = os.path.join(cnfg.OUTPUT_PATH, f"{exp_name}", f"{cnfg.SUBJECT_STR.capitalize()}_{subject_id}")
        if makedirs:
            os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, "Subject.pkl")

    def __repr__(self) -> str:
        return f"{self.experiment_name.upper()}-{cnfg.SUBJECT_STR.capitalize()}_{self.id}"
