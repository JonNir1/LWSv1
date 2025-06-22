from __future__ import annotations
import os
import time
import pickle as pkl
from typing import Union, Optional, List, Literal
from datetime import datetime

import numpy as np
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
        "Name": "name", "Age": "age", "Sex": "sex", "Handedness": "hand", "DominantEye": cnfg.EYE_STR,
        cnfg.SUBJECT_STR.capitalize(): f"{cnfg.SUBJECT_STR}_id",
        cnfg.SESSION_STR.capitalize(): cnfg.SESSION_STR,
        f"{cnfg.SESSION_STR.capitalize()}Date": f"{cnfg.SESSION_STR}_date",
        f"{cnfg.SESSION_STR.capitalize()}Time": f"{cnfg.SESSION_STR}_time",
        cnfg.DISTANCE_STR.capitalize(): "screen_distance_cm",
    }
    __EXTRACTION_FUNCTIONS = {
        cnfg.ACTION_STR: lambda trl: trl.get_actions(),
        cnfg.TARGET_STR: lambda trl: trl.get_target_identification_data(),
        cnfg.FIXATION_STR: lambda trl: trl.process_fixations(),
        cnfg.METADATA_STR: lambda trl: trl.get_metadata().to_frame().T.drop(columns=[f"{cnfg.TRIAL_STR}_num"]),
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
            1, "px", "deg", cnfg.PIXEL_SIZE_MM / 10, self._screen_distance_cm
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

    def get_target_identification_summary(self, verbose=False) -> pd.DataFrame:
        """
        Extracts the subject's targets-identification data and returns it as a DataFrame, indexed by a regular
        range-index, and with columns:
        - trial: int - trial number (has repetitions for each target)
        - trial_type: int - type of the trial (see LWSEnums.SearchArrayTypeEnum)
        - bad_trial: bool - whether the trial is "bad" (trial.is_bad() returns True)
        - trial_duration: float - duration of the trial in ms
        - target: str - target ID (e.g., "target_0", "target_1", etc.)
        - time: float - time of identification (in ms), or NaN if the target was not identified
        - distance_px, distance_dva: float - gaze distance from the target at the time of identification (in pixels and DVAs)
        - target_x, target_y, target_angle: target coordinates and rotation angle
        - target_category: int - category of the target (see LWSEnums.ImageCategoryEnum)
        """
        _COLUMNS_TO_DROP = [
            "target_sub_path", "left_x", "left_y", "left_pupil", "left_label", "right_x", "right_y", "right_pupil",
            "right_label", 'block_num', 'num_targets',
        ]
        targets = self._extract_df_from_trials(cnfg.TARGET_STR, verbose=verbose)
        metadata = self._extract_df_from_trials(cnfg.METADATA_STR, verbose=verbose)
        ident_data = pd.merge(targets, metadata, left_index=True, right_index=True, how='left')
        ident_data.reset_index(drop=False, inplace=True)
        ident_data.drop(columns=_COLUMNS_TO_DROP, inplace=True, errors='ignore')
        ident_data.rename(columns={"duration": f"{cnfg.TRIAL_STR}_duration"}, inplace=True)
        return ident_data

    def get_metadata(self, verbose: bool = False) -> pd.DataFrame:
        """
        Get the subject's trial metadata DataFrame for a subject.
        If the DataFrame is not found in the subject's output directory, it will be generated from the subject's trials.

        Returns a DataFrame containing metadata for each trial, indexed by trial number and with columns:
            - block_num: int - block number of the trial
            - trial_type: int (SearchArrayTypeEnum) - type of the trial (e.g., color, bw, etc.)
            - duration: float - duration of the trial in ms
            - num_targets: int - number of targets in the trial
            - bad_trial: bool - whether the trial is "bad", i,e, the subject performed "bad" actions during it (e.g. mark-and-reject)
        """
        return self._extract_df_from_trials(cnfg.METADATA_STR, verbose=verbose)

    def get_targets(self, verbose: bool = False) -> pd.DataFrame:
        """
        Get the target identification DataFrame for a subject.
        If the DataFrame is not found in the subject's output directory, it will be generated from the subject's trials.

        :param subject: Subject object containing trial data.
        :param save: If True, saves the DataFrame as a pickle file in the subject's output directory.
        :param verbose: If True, prints status messages during extraction.
        :return: DataFrame containing target identifications for each trial, indexed by trial number and with columns:
            - `time`: time of identification
            - `distance_px`: pixel-distance from the target at the time of identification
            - `left_x`, `left_y`: gaze coordinates in the left eye at the time of identification
            - `right_x`, `right_y`: gaze coordinates in the right eye at the time of identification
            - `left_pupil`, `right_pupil`: pupil size in the left/right eye at the time of identification
            - `'left_label'`, `'right_label'`: eye movement labels in the left/right eye at the time of identification
            - `target_x`, `target_y`: target coordinates
            - `target_angle`: target rotation angle
            - `target_sub_path`: path to the target image
            - `target_category`: target category
        """
        return self._extract_df_from_trials(cnfg.TARGET_STR, verbose=verbose)

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
            trial_triggers = triggers[triggers[cnfg.TRIAL_STR] == trial_num].copy()
            trial_gaze = gaze[gaze[cnfg.TRIAL_STR] == trial_num].copy()
            # adjust the time to start from 0
            min_time = np.nanmin([trial_triggers[cnfg.TIME_STR].min(), trial_gaze[cnfg.TIME_STR].min()])
            assert np.isfinite(min_time) and min_time >= 0, f"Start-time for trial {trial_num} is not valid: {min_time}"
            trial_triggers[cnfg.TIME_STR] -= min_time
            trial_gaze[cnfg.TIME_STR] -= min_time
            # create the Trial object
            trial = Trial(self, trial_triggers, trial_gaze)
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

    def _extract_df_from_trials(
            self, to_extract: Literal["action", "target", "metadata", "fixation"], verbose: bool = False
    ) -> pd.DataFrame:
        """ Iterates over the subject's trials and extracts the requested data type into a DataFrame. """
        to_extract = to_extract.lower()
        if to_extract not in self.__EXTRACTION_FUNCTIONS:
            raise ValueError(
                f"Unable to extract {to_extract}. Expected one of: "
                f"{cnfg.ACTION_STR}/{cnfg.TARGET_STR}/{cnfg.FIXATION_STR}/{cnfg.METADATA_STR}"
            )
        extraction_function = self.__EXTRACTION_FUNCTIONS[to_extract]
        dfs = dict()
        for trial in tqdm(self.get_trials(), desc=f"Extracting {to_extract.capitalize()}s", disable=not verbose):
            trial_df = extraction_function(trial)
            dfs[trial.trial_num] = trial_df
        df = pd.concat(dfs.values(), names=[cnfg.TRIAL_STR] + list(trial_df.index.names), keys=dfs.keys(), axis=0)

        # remove unnamed index levels if any exist
        index_names = pd.Series(df.index.names)
        if pd.isnull(index_names).any():
            df = df.reset_index(drop=False, inplace=False)
            df = df.set_index([name for name in index_names if pd.notnull(name)])
            df = df.drop(columns=[col for col in df.columns if col.startswith("level_")], inplace=False, errors='ignore')
        return df

    def __repr__(self) -> str:
        return f"{self.experiment_name.upper()}-{cnfg.SUBJECT_STR.capitalize()}_{self.id}"
