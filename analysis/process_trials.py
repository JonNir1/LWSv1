import os

import pandas as pd
from tqdm import tqdm

import config as cnfg
from data_models.Subject import Subject


def process_trials(subject: Subject, verbose: bool = False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Extracts the action, target, and fixation DataFrames for a given subject and saves them as pickle files.
    If the DataFrames already exist, they are read from the pickle files.

    DataFrames have the following structure:

    Actions: indexed by (trial number, action ID), with columns:
    - time: float - time of the action in ms
    - action: int (SearchActionTypesEnum) - type of the action

    Targets: indexed by (trial number, target ID), with columns:
    - x, y, angle: float - pixel coordinates and rotation-angle of the target
    - sub_path: str - path to the target image
    - category: int (ImageCategoryEnum) - category of the target image

    Fixations: indexed by (trial number, eye, fixation ID), with columns:
    - start-time, end-time, duration: float (in ms)
    - center-pixel: tuple (x, y) - the mean pixel coordinates of the fixation
    - pixel_std: tuple (x, y) - the standard deviation of the pixel coordinates of the fixation
    - outlier_reasons: List[str] - reasons for the fixation being an outlier (or [] if not an outlier)
    - target_0, target_1, ...: float - pixel-distances to each target in the trial
    - all_marked: List[str] - all targets that were identified previously or during the current fixation
    - curr_marked: str - the target that was identified during the current fixation (or None)
    - in_strip: bool - whether the fixation is in the bottom strip of the trial
    """
    actions = _read_or_extract(subject, cnfg.ACTION_STR, verbose=verbose)
    targets = _read_or_extract(subject, cnfg.TARGET_STR, verbose=verbose)
    fixations = _read_or_extract(subject, cnfg.FIXATION_STR, verbose=verbose)
    return actions, targets, fixations


def _read_or_extract(subject: Subject, descriptor: str, verbose: bool = False) -> pd.DataFrame:
    """
    Helper function to read or generate a DataFrame for a subject's trials based on the specified descriptor.
    Allowed descriptors are "action", "target", and "fixation"; yielding the following DataFrames:

    Action DataFrame: indexed by (trial number, action ID), containing the following columns:
    - time: float - time of the action in ms
    - action: int (SearchActionTypesEnum) - type of the action

    Target DataFrame: indexed by (trial number, target ID), containing the following columns:
    - x, y, angle: float - pixel coordinates and rotation-angle of the target
    - sub_path: str - path to the target image
    - category: int (ImageCategoryEnum) - category of the target image

    Fixation DataFrame: indexed by (trial number, eye, fixation ID), containing the following columns:
    - start-time, end-time and duration: float (in ms)
    - center-pixel: tuple (x, y) - the mean pixel coordinates of the fixation
    - pixel_std: tuple (x, y) - the standard deviation of the pixel coordinates of the fixation
    - outlier_reasons: List[str] - reasons for the fixation being an outlier (or [] if not and outlier)
    - target_0, target_1, ...: float - pixel-distances to each target in the trial
    - all_marked: List[str] - all targets that were identified previously or during the current fixation
    - curr_marked: str - the target that was identified during the current fixation (or None)
    - in_strip: bool - whether the fixation is in the bottom strip of the trial

    :raises ValueError: if descriptor does not match any of the expected values ("fixation", "action", "target").
    """
    descriptor = descriptor.lower()
    if descriptor == cnfg.ACTION_STR:
        extraction_function = lambda trl: trl.get_actions()
    elif descriptor == cnfg.TARGET_STR:
        extraction_function = lambda trl: trl.get_targets()
    elif descriptor == cnfg.FIXATION_STR:
        extraction_function = lambda trl: trl.process_fixations()
    else:
        raise ValueError(f"Unknown descriptor: {descriptor}. Expected one of: {cnfg.ACTION_STR}/{cnfg.TARGET_STR}/{cnfg.FIXATION_STR}")

    path = os.path.join(subject.out_dir, f'{descriptor}_df.pkl')
    try:
        df = pd.read_pickle(path)
        if verbose:
            print(f"{descriptor.capitalize()} DataFrame loaded.")
    except FileNotFoundError:
        if verbose:
            print(f"No {descriptor} DataFrame found. Extracting...")
        dfs = dict()
        for trial in tqdm(subject.get_trials(), desc=f"Extracting {descriptor.capitalize()}s", disable=not verbose):
            trial_df = extraction_function(trial)
            dfs[trial.trial_num] = trial_df
        df = pd.concat(dfs.values(), names=[cnfg.TRIAL_STR] + list(trial_df.index.names), keys=dfs.keys(), axis=0)
        df.to_pickle(path)
    return df
