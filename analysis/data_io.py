import os
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cnfg
from data_models.Subject import Subject
from data_models.LWSEnums import DominantEyeEnum

_DESCRIPTOR_MAP = {
    cnfg.ACTION_STR: lambda trl: trl.get_actions(),
    cnfg.TARGET_STR: lambda trl: trl.get_target_identification_data(),
    cnfg.FIXATION_STR: lambda trl: trl.process_fixations(),
    cnfg.METADATA_STR: lambda trl: trl.get_metadata().to_frame().T.drop(columns=[f"{cnfg.TRIAL_STR}_num"]),
}


def get_identifications(subject: Subject, save: bool = True, verbose: bool = False) -> pd.DataFrame:
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
    path = os.path.join(subject.out_dir, f'{cnfg.IDENTIFIED_STR}_df.pkl')
    try:
        ident_data = pd.read_pickle(path)
        if verbose:
            print(f"Identification DataFrame loaded.")
        return ident_data
    except FileNotFoundError:
        if verbose:
            print(f"No identification DataFrame found. Extracting...")
        # extract identification data from targets and metadata DataFrames
        targets_df = _read_or_extract(subject, cnfg.TARGET_STR, save=False, verbose=False)
        metadata_df = _read_or_extract(subject, cnfg.METADATA_STR, save=False, verbose=False)
        ident_data = pd.merge(
            targets_df[[cnfg.TIME_STR, f"{cnfg.TARGET_STR}_{cnfg.CATEGORY_STR}"]],
            metadata_df[[f"{cnfg.TRIAL_STR}_type", "is_bad"]],
            left_index=True, right_index=True, how='left'
        ).reset_index(drop=False)
        ident_data[cnfg.IDENTIFIED_STR] = np.isfinite(ident_data[cnfg.TIME_STR].values)
        ident_data.loc[~ident_data[cnfg.IDENTIFIED_STR].values, cnfg.TIME_STR] = np.nan    # set non-identified times to NaN

        # add fixation-related columns
        for eye in DominantEyeEnum:
            for col in ["", f"{cnfg.START_TIME_STR}", f"{cnfg.END_TIME_STR}", f"{cnfg.TARGET_STR}_{cnfg.DISTANCE_STR}"]:
                col_name = f"{eye}_{cnfg.FIXATION_STR}_{col}".strip("_")
                ident_data[col_name] = np.nan
        if save:
            ident_data.to_pickle(path)
        return ident_data


def get_fixations(subject: Subject, save: bool = True, verbose: bool = False) -> pd.DataFrame:
    """
    Get the fixations DataFrame for a subject.
    If the DataFrame is not found in the subject's output directory, it will be generated from the subject's trials.

    :param subject: Subject object containing trial data.
    :param save: If True, saves the DataFrame as a pickle file in the subject's output directory.
    :param verbose: If True, prints status messages during extraction.
    :return: DataFrame containing fixations for each trial, indexed by trial number and with columns:
        - start-time, end-time, duration: float (in ms)
        - center-pixel: tuple (x, y) - the mean pixel coordinates of the fixation
        - pixel_std: tuple (x, y) - the standard deviation of the pixel coordinates of the fixation
        - outlier_reasons: List[str] - reasons for the fixation being an outlier (or [] if not an outlier)
        - target_0, target_1, ...: float - pixel-distances to each target in the trial
        - all_marked: List[str] - all targets that were identified previously or during the current fixation
        - curr_marked: str - the target that was identified during the current fixation (or None)
        - in_strip: bool - whether the fixation is in the bottom strip of the trial
        - from_trial_start: float - time from trial's start to the start of the fixation (in ms)
        - to_trial_end: float - time from fixation's end to the end of the trial (in ms
    """
    return _read_or_extract(subject, cnfg.FIXATION_STR, save=save, verbose=verbose)


def get_visits(
        subject: Subject,
        distance_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
        save: bool = True,
        verbose: bool = False
) -> pd.DataFrame:
    """
    Get the visits DataFrame for a subject.
    If the DataFrame is not found in the subject's output directory, it will be generated from the subject's fixations.

    :param subject: Subject object containing trial data.
    :param distance_threshold_dva: float - the distance threshold in degrees of visual angle (DVA) to consider a fixation as "on target".
    :param save: If True, saves the DataFrame as a pickle file in the subject's output directory.
    :param verbose: If True, prints status messages during extraction.
    :return: DataFrame containing visits for each trial, indexed by (trial, target, eye, visit) and with columns:
        - start_fixation, end_fixation: int - the start and end fixation IDs of the visit
        - start_time, end_time, duration: float - the start, end, and duration of the visit (in ms)
        - num_fixations: int - the number of fixations in the visit
        - min_distance, max_distance: float - the minimum and maximum distance to the target during the visit (in pixels)
    """
    distance_threshold_dva = round(distance_threshold_dva, 1)
    assert distance_threshold_dva >= 0, "Distance threshold must be non-negative."
    path = os.path.join(subject.out_dir, f"{cnfg.VISIT_STR}_df_{distance_threshold_dva:.1f}DVA.pkl")
    try:
        visits_df = pd.read_pickle(path)
        if verbose:
            print(f"Visits DataFrame loaded.")
        return visits_df
    except FileNotFoundError:
        if verbose:
            print(f"No visits DataFrame found. Extracting...")
        visits_df = _extract_visits(
            subject, distance_threshold_dva, get_fixations(subject, save=False, verbose=False)
        )
        if save:
            visits_df.to_pickle(path)
        return visits_df


def get_metadata(subject: Subject, save: bool = True, verbose: bool = False) -> pd.DataFrame:
    """
    Get the trials' metadata DataFrame for a subject.
    If the DataFrame is not found in the subject's output directory, it will be generated from the subject's trials.

    :param subject: Subject object containing trial data.
    :param save: If True, saves the DataFrame as a pickle file in the subject's output directory.
    :param verbose: If True, prints status messages during extraction.
    :return: DataFrame containing metadata for each trial, indexed by trial number and with columns:
        - block_num: int - block number of the trial
        - trial_type: int (SearchArrayTypeEnum) - type of the trial (e.g., color, bw, etc.)
        - duration: float - duration of the trial in ms
        - num_targets: int - number of targets in the trial
        - is_bad: bool - whether the trial is "bad", i,e, the subject performed "bad" actions during it (e.g. mark-and-reject)
    """
    return _read_or_extract(subject, cnfg.METADATA_STR, save=save, verbose=verbose)


def get_actions(subject: Subject, save: bool = True, verbose: bool = False) -> pd.DataFrame:
    """
    Get the actions DataFrame for a subject.
    If the DataFrame is not found in the subject's output directory, it will be generated from the subject's trials.

    :param subject: Subject object containing trial data.
    :param save: If True, saves the DataFrame as a pickle file in the subject's output directory.
    :param verbose: If True, prints status messages during extraction.
    :return: DataFrame containing actions for each trial, indexed by trial number and with columns:
        - action_type: int (SubjectActionTypesEnum) - type of action performed
        - target_num: int - target number (if applicable)
        - time: float - time of the action from trial's start, in ms
    """
    return _read_or_extract(subject, cnfg.ACTION_STR, save=save, verbose=verbose)


def get_targets(subject: Subject, save: bool = True, verbose: bool = False) -> pd.DataFrame:
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
    return _read_or_extract(subject, cnfg.TARGET_STR, save=save, verbose=verbose)


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
            "Unknown descriptor: {descriptor}. Expected one of: "
            f"{cnfg.ACTION_STR}/{cnfg.TARGET_STR}/{cnfg.FIXATION_STR}/{cnfg.METADATA_STR}"
        )
    extraction_function = _DESCRIPTOR_MAP[descriptor]
    path = os.path.join(subject.out_dir, f'{descriptor}_df.pkl')
    try:
        df = pd.read_pickle(path)
        if verbose:
            print(f"{descriptor.capitalize()} DataFrame loaded.")
        return df
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


def _extract_visits(
        subject: Subject,
        distance_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
        fixations_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Converts the fixations DataFrame into a `visits` DataFrame for a given subject. A `visit` is defined as a sequence
    of consecutive fixations that are within a specified distance threshold (in degrees of visual angle) from the same
    target.

    The DataFrame is indexed by (trial, target, eye, visit), with columns:
    - start_fixation, end_fixation: int - the start and end fixation IDs of the visit
    - start_time, end_time, duration: float - the start, end, and duration of the visit (in ms)
    - num_fixations: int - the number of fixations in the visit
    - min_distance, max_distance: float - the minimum and maximum distance to the target during the visit (in pixels)
    """
    fixations_df = fixations_df or get_fixations(subject, save=False, verbose=False)
    visits = []
    target_cols = [col for col in fixations_df.columns if col.startswith(f"{cnfg.TARGET_STR}_")]
    for (trial, eye), group in fixations_df.groupby(level=[cnfg.TRIAL_STR, cnfg.EYE_STR]):
        group = group.reset_index().sort_values(cnfg.FIXATION_STR, inplace=False)
        for target in target_cols:
            on_target = group[target] < distance_threshold_dva / subject.px2deg
            visit_id = (on_target != on_target.shift()).cumsum()
            group["visit_group"] = np.where(on_target, visit_id, np.nan)
            for visit_idx in group["visit_group"].dropna().unique():
                visit_fixs = group[group["visit_group"] == visit_idx]
                if visit_fixs.empty:
                    continue
                visits.append({
                    cnfg.TRIAL_STR: trial,
                    cnfg.EYE_STR: eye,
                    cnfg.TARGET_STR: target,
                    cnfg.VISIT_STR: int(visit_idx),
                    f"start_{cnfg.FIXATION_STR}": int(visit_fixs[cnfg.FIXATION_STR].iloc[0]),
                    f"end_{cnfg.FIXATION_STR}": int(visit_fixs[cnfg.FIXATION_STR].iloc[-1]),
                    cnfg.START_TIME_STR: visit_fixs[cnfg.START_TIME_STR].iloc[0],
                    cnfg.END_TIME_STR: visit_fixs[cnfg.END_TIME_STR].iloc[-1],
                    "duration": visit_fixs[cnfg.END_TIME_STR].iloc[-1] - visit_fixs[cnfg.START_TIME_STR].iloc[0],
                    "num_fixations": len(visit_fixs),
                    f"min_{cnfg.DISTANCE_STR}": visit_fixs[target].min(),
                    f"max_{cnfg.DISTANCE_STR}": visit_fixs[target].max(),
                })
    visits_df = pd.DataFrame(visits)
    visits_df.set_index([cnfg.TRIAL_STR, cnfg.TARGET_STR, cnfg.EYE_STR, cnfg.VISIT_STR], inplace=True)
    return visits_df
