import os

import pandas as pd
from tqdm import tqdm

import config as cnfg
from data_models.Subject import Subject


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
    - bad_actions: bool - whether the subject performed any "bad" actions during the trial (e.g. mark-and-reject)

    Actions: indexed by trial number, with columns:
    - time: float - time of the action in ms
    - action: int (SearchActionTypesEnum) - type of the action

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
    """
    metadata = _read_or_extract(subject, cnfg.METADATA_STR, save=save, verbose=verbose).droplevel(1)
    actions = _read_or_extract(subject, cnfg.ACTION_STR, save=save, verbose=verbose).droplevel(1)
    targets = _read_or_extract(subject, cnfg.TARGET_STR, save=save, verbose=verbose)
    fixations = _read_or_extract(subject, cnfg.FIXATION_STR, save=save, verbose=verbose)
    return metadata, actions, targets, fixations


def _read_or_extract(subject: Subject, descriptor: str, save: bool = True, verbose: bool = False) -> pd.DataFrame:
    """
    Helper function to read or generate a DataFrame for a subject's trials based on the specified descriptor.
    If the `save` parameter is True, the DataFrame will be saved as a pickle file in the subject's output directory.
    Allowed descriptors are "action", "target", "fixation", and "metadata".

    :raises ValueError: if descriptor does not match any of the allowed values.
    """
    descriptor = descriptor.lower()
    if descriptor == cnfg.ACTION_STR:
        extraction_function = lambda trl: trl.get_actions()
    elif descriptor == cnfg.TARGET_STR:
        extraction_function = lambda trl: trl.get_target_identification_data()
    elif descriptor == cnfg.FIXATION_STR:
        extraction_function = lambda trl: trl.process_fixations()
    elif descriptor == cnfg.METADATA_STR:
        extraction_function = lambda trl: trl.get_metadata().to_frame().T.drop(columns=[f"{cnfg.TRIAL_STR}_num"])
    else:
        raise ValueError(
            f"Unknown descriptor: {descriptor}. Expected one of: {cnfg.ACTION_STR}/{cnfg.TARGET_STR}/{cnfg.FIXATION_STR}/{cnfg.METADATA_STR}"
        )

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
        if save:
            df.to_pickle(path)
    return df
