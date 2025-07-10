from __future__ import annotations
import os
import time
import pickle as pkl
from typing import Union, Optional, List, Sequence
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cnfg
import helpers as hlp
from data_models.io_helpers.subject_info import parse_subject_info
from data_models.io_helpers.triggers_and_gaze import parse_triggers_and_gaze
from data_models.io_helpers.target_identifications import extract_trial_identifications
from data_models.io_helpers.visits import convert_fixations_to_visits
from data_models.LWSEnums import SexEnum, DominantHandEnum, DominantEyeEnum, SubjectActionCategoryEnum


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
            print(f"Completed parsing subject in {time.time() - start:.2f} seconds.")
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

    def get_targets(self) -> pd.DataFrame:
        """
        Extract the target information from a subject's trials, returning a DataFrame with the following columns:
        - trial: int; the trial number
        - target: str; the name of the target
        - x: float; the x coordinate of the target in pixels
        - y: float; the y coordinate of the target in pixels
        - angle: float; the rotation angle of the target in degrees
        - category: ImageCategoryEnum; the category of the target
        - sub_path: str; the path to the target image file, relative to the images directory
        """
        targets = dict()
        for trial in tqdm(self.get_trials(), desc="Extracting Targets", disable=True):
            targets[trial.trial_num] = (
                trial.get_targets()
                .rename(columns=lambda name: name.replace(f"{cnfg.TARGET_STR}_", ""))
                .reset_index(drop=False)
                .rename(columns={"index": cnfg.TARGET_STR, })
                .sort_values(by=cnfg.TARGET_STR)
            )
        targets = (
            pd.concat(targets.values(), axis=0, keys=targets.keys())
            .reset_index(drop=False)
            .rename(columns={"level_0": cnfg.TRIAL_STR})
            .drop(columns=["level_1"])
        )
        return targets

    def get_actions(self) -> pd.DataFrame:
        actions = dict()
        for trial in tqdm(self.get_trials(), desc="Extracting Actions", disable=True):
            actions[trial.trial_num] = trial.get_actions()
        actions = pd.concat(actions.values(), axis=0, keys=actions.keys())
        actions = (
            actions
            .reset_index(drop=False)
            .rename(columns={"level_0": cnfg.TRIAL_STR})
            .drop(columns=["level_1"])
        )
        return actions


    def get_metadata(self, bad_actions: Sequence[SubjectActionCategoryEnum]) -> pd.DataFrame:
        """
        Extract the subject's trial metadata into a DataFrame, containing the following columns:
        - trial_num
        - block_num
        - trial_category: COLOR/BW/NOISE
        - duration: in ms
        - num_targets
        - bad_actions: boolean; True if any of the subject's actions during the trial are considered bad
        """
        metadata = dict()
        for trial in tqdm(self.get_trials(), desc="Trial Metadata", disable=True):
            metadata[trial.trial_num] = trial.get_metadata(bad_actions)
        res = pd.concat(metadata.values(), keys=metadata.keys(), axis=1).T
        res = res.reset_index(drop=True)
        return res

    def get_target_identifications(
            self,
            identification_actions: Union[Sequence[SubjectActionCategoryEnum], SubjectActionCategoryEnum],
            temporal_matching_threshold: float,
            on_target_threshold_dva: float,
            verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Extracts the target identification behavior of a subject across all trials.
        :param identification_actions: action(s) that indicate the subject has identified a target.
        :param temporal_matching_threshold: temporal threshold (in ms) for matching gaze samples to identification actions.
        :param on_target_threshold_dva: the distance in DVA from the target to consider the identification as a hit.
        :param verbose: if True, displays a progress bar for the extraction process.

        :return: a DataFrame containing the target identification behavior for each trial, with the following columns:
        - trial: int; the trial number
        - target: str; the name of the closest target to the subject's gaze at the time of identification
        - time: float; the time of the identification action in ms (relative to trial onset)
        - distance_px: float; the distance between the subject's gaze and the closest target, in pixels
        - distance_dva: float; the distance between the subject's gaze and the closest target, in DVA
        - left_x, left_y, right_x, right_y: float; the x and y coordinates of the subject's left and right eye gaze at the time of identification
        - left_pupil, right_pupil: float; the pupil size of the subject's left and right eye at the time of identification
        """
        trial_idents = dict()
        for trial in tqdm(self.get_trials(), desc="Target Identifications", disable=not verbose):
            trial_idents[trial.trial_num] = extract_trial_identifications(
                trial=trial,
                identification_actions=identification_actions,
                gaze_to_trigger_matching_threshold=temporal_matching_threshold,
                on_target_threshold_dva=on_target_threshold_dva,
            )
        idents = pd.concat(trial_idents.values(), axis=0, keys=trial_idents.keys())
        idents = (
            idents
            .reset_index(drop=False)
            .drop(
                columns=["target_sub_path", "level_1", "left_label", "right_label", ],
                inplace=False,
                errors='ignore'
            )
            .rename(columns={"level_0": cnfg.TRIAL_STR})
            .sort_values(by=[cnfg.TRIAL_STR, cnfg.TARGET_STR])
            .reset_index(drop=True)
        )
        return idents

    def get_fixations(self, save: bool = True, verbose: bool = False,) -> pd.DataFrame:
        """
        Extracts the subject's fixations across all trials and returns them as a DataFrame.
        :param save: bool; if True, saves the fixations DataFrame to a pickle file in the subject's output directory.
        :param verbose: bool; if True, displays a progress bar for the extraction process and prints messages about the process.

        :return: a DataFrame containing the fixations for each trial, with the following columns:
        - trial: int; the trial number
        - eye: str; the eye that the fixation belongs to (left or right)
        - event: int; the number of the fixation among all events from the given eye during the trial
        - start_time: float; time of the fixation start in ms (relative to trial onset)
        - end_time: float; time of the fixation end in ms (relative to trial onset)
        - duration: float; duration of the fixation in ms
        - to_trial_end: float; time from the end of the fixation to the end of the trial in ms
        - x: float; x coordinates of the fixation center in pixels
        - y: float; y coordinates of the fixation center in pixels
        - outlier_reasons: List[str]; reasons for the fixation to be an outlier
        - target: str; the name of the closest target to the fixation center at the time of the fixation
        - target{i}_distance_px: float; the distance between the fixation center and target{i} in pixels (target0, target1, etc.)
        - target{i}_distance_dva: float; the distance between the fixation center and target{i} in DVA (target0, target1, etc.)
        - num_fixs_to_strip: int; number of fixations from the current fixation until a visit in the bottom strip of the
        SearchArray. Value is 0 if he current fixation is in the bottom strip, and np.inf if there are no future fixations
        in the strip during the trial.
        """
        path = os.path.join(self.out_dir, f'{cnfg.FIXATION_STR}_df.pkl')
        try:
            fixations = pd.read_pickle(path)
            if verbose:
                print(f"Subject {self.id}'s fixations DataFrame loaded.")
        except FileNotFoundError:
            if verbose:
                print(f"Fixations DataFrame not found for subject {self.id}. Extracting...")
            fixations = self._process_fixations(verbose)
            if save:
                fixations.to_pickle(path)
        return fixations

    def get_visits(self, target_distance_threshold_dva: float, visit_merging_time_threshold: float,) -> pd.DataFrame:
        fixations = self.get_fixations(save=False, verbose=False)
        return convert_fixations_to_visits(
            fixations,
            target_distance_threshold_dva,
            visit_merging_time_threshold,
        )

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
        out_dir = os.path.join(cnfg.SUBJECT_OUTPUT_PATH, f"{exp_name}_{cnfg.SUBJECT_STR.capitalize()}_{subject_id:02d}")
        if makedirs:
            os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, "Subject.pkl")

    def _process_fixations(self, verbose: bool = True) -> pd.DataFrame:
        trial_fixations = dict()
        for trial in tqdm(self.get_trials(), desc=f"Extracting Fixations", disable=not verbose):
            trial_fixations[trial.trial_num] = trial.process_fixations()
        fixations = pd.concat(trial_fixations.values(), axis=0, keys=trial_fixations.keys())
        fixations = (
            fixations
            .reset_index(drop=False)
            .drop(columns=["level_1"])
            .rename(columns={"level_0": cnfg.TRIAL_STR})
        )
        return fixations

    def __repr__(self) -> str:
        return f"{self.experiment_name.upper()}-{cnfg.SUBJECT_STR.capitalize()}_{self.id}"
