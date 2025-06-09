import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cnfg
from data_models.Subject import Subject

_DESCRIPTOR_MAP = {
    cnfg.ACTION_STR: lambda trl: trl.get_actions(),
    cnfg.TARGET_STR: lambda trl: trl.get_target_identification_data(),
    cnfg.FIXATION_STR: lambda trl: trl.process_fixations(),
    cnfg.METADATA_STR: lambda trl: trl.get_metadata().to_frame().T.drop(columns=[f"{cnfg.TRIAL_STR}_num"]),
}


def process_trials(
        subject: Subject, save: bool = True, verbose: bool = False
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Extracts the action, target, and fixation DataFrames for a given subject and saves them as pickle files.
    If the DataFrames already exist, they are read from the pickle files. If not, they are generated from the subject's
    trials and saved to the subject's output directory if `save` is True.

    DataFrames have the following structure:

    Metadata: indexed by trial number, with columns:
    - block_num: int - block number of the trial
    - trial_type: int (SearchArrayTypeEnum) - type of the trial (e.g., color, bw, etc.)
    - duration: float - duration of the trial in ms
    - num_targets: int - number of targets in the trial
    - is_bad: bool - whether the trial is "bad", i,e, the subject performed "bad" actions during it (e.g. mark-and-reject)

    Actions: indexed by trial number, with columns:
    - time: float - time of the action in ms
    - action: int (SubjectActionTypesEnum) - type of the action

    Targets: indexed by (trial number, target ID), with columns:
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

    Fixations: indexed by (trial number, eye, fixation ID), with columns:
    - start-time, end-time, duration: float (in ms)
    - center-pixel: tuple (x, y) - the mean pixel coordinates of the fixation
    - pixel_std: tuple (x, y) - the standard deviation of the pixel coordinates of the fixation
    - outlier_reasons: List[str] - reasons for the fixation being an outlier (or [] if not an outlier)
    - target_0, target_1, ...: float - pixel-distances to each target in the trial
    - all_marked: List[str] - all targets that were identified previously or during the current fixation
    - curr_marked: str - the target that was identified during the current fixation (or None)
    - in_strip: bool - whether the fixation is in the bottom strip of the trial
    - from_trial_start: float - time from trial's start to the start of the fixation (in ms)
    - to_trial_end: float - time from fixation's end to the end of the trial (in ms)
    """
    metadata, actions, targets, fixations =  tuple(
        _read_or_extract(subject, desc, save=save, verbose=verbose)
        for desc in [cnfg.METADATA_STR, cnfg.ACTION_STR, cnfg.TARGET_STR, cnfg.FIXATION_STR]
    )
    return metadata, actions, targets, fixations


def identification_data(subject: Subject, save: bool = True, verbose: bool = False) -> pd.DataFrame:
    """
    Extracts the identification DataFrame for a given subject and saves it as a pickle file.
    If the DataFrame already exists, it is read from the pickle file. If not, it is generated from the subject's
    trials and saved to the subject's output directory if `save` is True.

    The DataFrame is indexed by a regular range-index, with columns:
    - trial: int - trial number (has repetitions for each target)
    - target: str - target ID (e.g., "target_0", "target_1", etc.)
    - time: float - time of identification (in ms)
    - target_category: int - category of the target (see LWSEnums.ImageCategoryEnum)
    - trial_type: int - type of the trial (see LWSEnums.SearchArrayTypeEnum)
    - is_bad: bool - whether the trial is "bad", i.e., the subject performed "bad" actions during the trial (e.g. mark-and-reject)
    - identified: bool - whether the target was identified in the trial (True if `time` is finite, False otherwise)
    """
    # TODO: add fixation start-time and time from trial start, for the identification fixation
    path = os.path.join(subject.out_dir, f'{cnfg.IDENTIFIED_STR}_df.pkl')
    try:
        ident_data = pd.read_pickle(path)
    except FileNotFoundError:
        targets_df = _read_or_extract(subject, cnfg.TARGET_STR, save=False, verbose=False)
        metadata_df = _read_or_extract(subject, cnfg.METADATA_STR, save=False, verbose=False)
        ident_data = pd.merge(
            targets_df[[cnfg.TIME_STR, f"{cnfg.TARGET_STR}_{cnfg.CATEGORY_STR}"]],
            metadata_df[[f"{cnfg.TRIAL_STR}_type", "is_bad"]],
            left_index=True, right_index=True, how='left'
        ).reset_index(drop=False)
        ident_data[cnfg.IDENTIFIED_STR] = np.isfinite(ident_data[cnfg.TIME_STR].values)
        ident_data.loc[~ident_data[cnfg.IDENTIFIED_STR].values, cnfg.TIME_STR] = np.nan    # set non-identified times to NaN
    if save:
        ident_data.to_pickle(path)
    return ident_data


def fixation_data(subject: Subject, save: bool = True, verbose: bool = False) -> pd.DataFrame:
    """
    Extracts the fixation DataFrame for a given subject and saves it as a pickle file.
    If the DataFrame already exists, it is read from the pickle file. If not, it is generated from the subject's
    trials and saved to the subject's output directory if `save` is True.

    The DataFrame is indexed by (trial number, eye, fixation ID), with columns:
    - start-time, end-time, duration: float (in ms)
    - center-pixel: tuple (x, y) - the mean pixel coordinates of the fixation
    - pixel_std: tuple (x, y) - the standard deviation of the pixel coordinates of the fixation
    - outlier_reasons: List[str] - reasons for the fixation being an outlier (or [] if not an outlier)
    - target_0, target_1, ...: float - pixel-distances to each target in the trial
    - all_marked: List[str] - all targets that were identified previously or during the current fixation
    - curr_marked: str - the target that was identified during the current fixation (or None)
    - in_strip: bool - whether the fixation is in the bottom strip of the trial
    - from_trial_start: float - time from trial's start to the start of the fixation (in ms)
    - to_trial_end: float - time from fixation's end to the end of the trial (in ms)
    """
    return _read_or_extract(subject, cnfg.FIXATION_STR, save=save, verbose=verbose)


def _read_or_extract(subject: Subject, descriptor: str, save: bool = True, verbose: bool = False) -> pd.DataFrame:
    """
    Helper function to read or generate a DataFrame for a subject's trials based on the specified descriptor.
    If the `save` parameter is True, the DataFrame will be saved as a pickle file in the subject's output directory.
    Allowed descriptors are "action", "target", "fixation", and "metadata".

    :raises ValueError: if descriptor does not match any of the allowed values.
    """
    descriptor = descriptor.lower()
    if descriptor not in _DESCRIPTOR_MAP:
        raise ValueError(
            f"Unknown descriptor: {descriptor}. Expected one of: {cnfg.ACTION_STR}/{cnfg.TARGET_STR}/{cnfg.FIXATION_STR}/{cnfg.METADATA_STR}"
        )
    extraction_function = _DESCRIPTOR_MAP[descriptor]
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

        # remove unnamed index levels if any
        index_names = pd.Series(df.index.names)
        if pd.isnull(index_names).any():
            # if any of the index names are NaN, remove those levels from the dataframe
            df = df.reset_index(drop=False, inplace=False)
            df = df.set_index([name for name in index_names if pd.notnull(name)])
            df = df.drop(columns=[col for col in df.columns if col.startswith("level_")], inplace=False, errors='ignore')
    if save:
        df.to_pickle(path)
    return df
