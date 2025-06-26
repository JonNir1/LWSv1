import os
import time
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import peyes

import config as cnfg
import helpers as hlp
from data_models.Subject import Subject
from data_models.SearchArray import SearchArray

_FIXATION_LABEL = peyes.parse_label(cnfg.FIXATION_STR)
_REDUNDANT_FIXATION_FEATURES = [
    'label', "center_pixel", "pixel_std", 'azimuth', 'is_outlier',
    "distance", "cumulative_distance", "amplitude", "cumulative_amplitude",
    'velocity', "peak_velocity", "median_velocity", "min_velocity",
    'dispersion', "area", "median_dispersion", "ellipse_area",
]


def get_fixations(subject: Subject, save: bool = True, verbose: bool = False) -> pd.DataFrame:
    """
    Get the fixations DataFrame for a subject.
    If the DataFrame is not found in the subject's output directory, it will be generated from the subject's trials and
    saved to the subject's output directory if `save` is True. If `include_outliers` is False, outlier fixations are
    filtered out from the DataFrame.

    :param subject: Subject object containing trial data.
    :param save: If True, saves the DataFrame as a pickle file in the subject's output directory.
    :param verbose: If True, prints status messages during extraction.
    :return: DataFrame containing fixations for each trial, indexed by trial number and with columns:
    - trial: int; the trial number in which the fixation was detected
    - eye: str; the eye on which the fixation was detected ("left" or "right")
    - fixation: int; the fixation number within the (trial, eye) pair
    - start-time, end-time: float (in ms); relative to the trial start time
    - duration: float (in ms); the duration of the fixation
    - to_trial_end: float; time from the end of the fixation to the end of the trial (in ms)
    - outlier_reasons: List[str] - reasons for the fixation being an outlier (or [] if not an outlier)
    - x, y: float; mean pixel coordinates of the fixation
    - target: str (target_0, target_1, ...); name of closest target to the fixation
    - distance: float; distance to the closest target (in DVA)
    - num_fixs_to_strip: float; number of fixations until the next visit to the SearchArray bottom strip, or inf if no such visit exists
    - curr_identified: str - the target that was identified during the current fixation (or None)
    - target_time: float - the time of the target identification in the trial, or inf if the target was never identified
    - target_x, target_y, target_angle: float - the pixel coordinates and angle of the target
    - target_category: int - the category of the target (see ImageCategoryEnum)
    - trial_type: int - the type of the trial (e.g., "color", "bw", "noise"; see SearchArrayTypeEnum)
    - trial_duration: float - the duration of the trial (in ms)
    - bad_trial: bool - whether the trial is considered "bad" (e.g., if the subject performed a "bad" action)
    """
    path = os.path.join(subject.out_dir, f'{cnfg.FIXATION_STR}_df.pkl')
    try:
        df = pd.read_pickle(path)
        if verbose:
            print(f"{cnfg.FIXATION_STR.capitalize()} DataFrame loaded.")
        return df
    except FileNotFoundError:
        if verbose:
            print(f"No {cnfg.FIXATION_STR} DataFrame found. Extracting...")
        fixations_df = extract_fixations(subject, verbose=verbose)
        if save:
            fixations_df.to_pickle(path)
        return fixations_df


def extract_fixations(subject: Subject, verbose: bool = False) -> pd.DataFrame:
    start = time.time()
    fixations_df = _extract_fixation_df(subject, verbose=verbose)
    if verbose:
        print(f"Appending target information to {fixations_df.shape[0]} fixations...")
    idents_df = subject.get_target_identification_summary()
    all_distances = _calc_target_distances(fixations_df, idents_df)
    closest_dists = _find_closest_target(all_distances)
    currently_identifying = _currently_identifying(fixations_df, idents_df)
    to_strip = _num_fixations_to_strip(fixations_df)
    fixations_df = pd.concat([fixations_df, closest_dists, currently_identifying, to_strip], axis=1)
    fixations_df[cnfg.DISTANCE_STR] *= subject.px2deg   # convert pixel distances to DVA

    # merge with target identification data
    fixations_df = fixations_df.set_index([cnfg.TRIAL_STR, cnfg.TARGET_STR])
    targets_df = subject.get_target_identification_summary(verbose=False)
    targets_df = (
        targets_df
        .set_index([cnfg.TRIAL_STR, cnfg.TARGET_STR])
        .rename(columns={cnfg.TIME_STR: cnfg.TARGET_TIME_STR})
        .drop(columns=[col for col in targets_df.columns if col.startswith(cnfg.DISTANCE_STR)], errors='ignore')
        .loc[fixations_df.index]
    )
    merged = (
        pd.concat([fixations_df, targets_df], axis=1)
        .reset_index(drop=False)
        .sort_values([cnfg.TRIAL_STR, cnfg.TARGET_STR, cnfg.EYE_STR, cnfg.FIXATION_STR], inplace=False)
        .reset_index(drop=True)
    )
    if verbose:
        print(f"Extracted fixations in {time.time() - start:.2f} seconds.")
    return merged


def _extract_fixation_df(subject: Subject, verbose: bool = False) -> pd.DataFrame:
    """
    Extracts all fixations detected during the subject's trials, and summarizes them into a DataFrame with the
    following columns:
    - start_time, end_time: float; fixation start-time relative to the trial start (in ms)
    - duration: float; fixation duration (in ms)
    - to_trial_end: float; time from the end of the fixation to the end of the trial (in ms)
    - x, y: float; mean pixel coordinates of the fixation
    - outlier_reasons: List[str]; reasons for the fixation being an outlier (or [] if not an outlier)
    - trial: int; the trial number in which the fixation was detected
    - eye: str; the eye on which the fixation was detected ("left" or "right")
    - fixation: int; the fixation number within the trial
    """
    dfs = dict()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for trial in tqdm(subject.get_trials(), desc=f"Extracting {cnfg.FIXATION_STR}s", disable=not verbose):
            trial_df = trial.get_eye_movements()
            trial_df = trial_df[trial_df[cnfg.LABEL_STR] == _FIXATION_LABEL]
            trial_df["to_trial_end"] = trial.end_time - trial_df[cnfg.END_TIME_STR]
            dfs[trial.trial_num] = trial_df
    fixation_df = pd.concat(
        dfs.values(), names=[cnfg.TRIAL_STR, cnfg.EYE_STR, cnfg.FIXATION_STR], keys=dfs.keys(), axis=0
    )
    centers = pd.DataFrame(
        fixation_df["center_pixel"].to_list(), index=fixation_df.index, columns=[cnfg.X, cnfg.Y]
    )
    fixation_df = pd.concat([fixation_df, centers], axis=1)
    fixation_df.drop(columns=_REDUNDANT_FIXATION_FEATURES, inplace=True, errors='ignore')
    fixation_df.reset_index(drop=False, inplace=True)
    return fixation_df


def _calc_target_distances(fixations_df: pd.DataFrame, idents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the pixel distances from each fixation to each target in the trial.
    The resulting DataFrame will have a column for each target, named as "target_0", "target_1", etc.
    """
    dists = dict()
    for trial_num, trial_fixs in fixations_df.groupby(cnfg.TRIAL_STR):
        trial_targets = idents_df[idents_df[cnfg.TRIAL_STR] == trial_num]
        if trial_targets.empty:
            continue
        # calculate distances
        fixs_xy = trial_fixs[[cnfg.X, cnfg.Y]].to_numpy()   # shape (num_fixations, 2)
        tgt_xy = trial_targets[                             # shape (num_targets, 2)
            [f"{cnfg.TARGET_STR}_{cnfg.X}", f"{cnfg.TARGET_STR}_{cnfg.Y}"]
        ].to_numpy()
        dxy = fixs_xy[:, None, :] - tgt_xy[None, :, :]      # shape (num_fixations, num_targets, 2)
        distances = np.linalg.norm(dxy, axis=2)             # shape (num_fixations, num_targets)
        # create a DataFrame with distances
        distances = pd.DataFrame(distances, columns=trial_targets[cnfg.TARGET_STR].to_list(), index=trial_fixs.index)
        dists[trial_num] = distances
    dists_df = pd.concat(dists, axis=0).reset_index(drop=True)
    return dists_df


def _find_closest_target(distances: pd.DataFrame) -> pd.DataFrame:
    shortest_distance = distances.min(axis=1).rename(cnfg.DISTANCE_STR)
    closest_target = distances.idxmin(axis=1).rename(cnfg.TARGET_STR)
    closest_target_df = pd.concat([shortest_distance, closest_target], axis=1)
    return closest_target_df


# TODO: add `num_fixs_to_identification` to fixs_df (instead of `next_1/2/3_in_strip`)
def _currently_identifying(fixations_df: pd.DataFrame, idents_df: pd.DataFrame) -> pd.Series:
    """ For each fixation in the fixations DataFrame, determine which target was currently being identified, if any. """
    current = dict()
    for trial_num, trial_fixs in fixations_df.groupby(cnfg.TRIAL_STR):
        trial_targets = idents_df[idents_df[cnfg.TRIAL_STR] == trial_num]
        if trial_targets.empty:
            continue
        tgt_times = trial_targets[cnfg.TIME_STR].to_numpy()     # shape (num_targets,)
        is_start_before = trial_fixs[cnfg.START_TIME_STR].to_numpy() <= tgt_times[:, None]  # shape (num_targets, num_fixations)
        is_end_after = trial_fixs[cnfg.END_TIME_STR].to_numpy() >= tgt_times[:, None]       # shape (num_targets, num_fixations)
        is_currently_identifying = is_start_before & is_end_after                           # shape (num_targets, num_fixations)
        is_currently_identifying = pd.DataFrame(
            is_currently_identifying.T, index=trial_fixs.index, columns=trial_targets[cnfg.TARGET_STR].to_list()
        )

        if not is_currently_identifying[is_currently_identifying.sum(axis=1) > 1].empty:
            # TODO: consider resolving this in code rather than manually?
            warnings.warn(
                f"Multiple targets marked during the same fixation in trial {trial_num}. "
                "This is not expected and may indicate an error in the data.",
                RuntimeWarning,
            )
        identified = is_currently_identifying[is_currently_identifying.any(axis=1)].idxmax(axis=1)  # pd.Series; len num_marking_fixs
        curr_ident = pd.Series(None, index=trial_fixs.index, name="curr_identified", dtype=str)
        curr_ident.loc[identified.index] = identified.values  # set the currently marked targets for marking fixations
        current[trial_num] = curr_ident
    current = pd.concat(current, axis=0).reset_index(drop=True)
    current.name = "curr_identified"
    return current


def _num_fixations_to_strip(fixations_df: pd.DataFrame) -> pd.Series:
    """ For each fixation, checks if it is in the bottom strip of the SearchArray. """
    fixations_copy = fixations_df.copy()
    xy = fixations_df[[cnfg.X, cnfg.Y]]      # shape (num_fixs, 2)
    fixations_copy['in_strip'] = pd.Series(
        map(lambda tup: SearchArray.is_in_bottom_strip((tup.x, tup.y)), xy.itertuples()),
        name="in_strip", dtype=bool,
    )
    result_parts = []
    for (_trial, _eye), group in fixations_copy.groupby(by=[cnfg.TRIAL_STR, cnfg.EYE_STR]):
        group = group.sort_values(cnfg.FIXATION_STR, inplace=False)
        num_fixs_to_strip = hlp.num_to_true(group["in_strip"])
        result_parts.append(num_fixs_to_strip)
    # concatenate results and return
    num_fixs_to_strip = pd.concat(result_parts).reset_index(drop=True)
    num_fixs_to_strip.name = "num_fixs_to_strip"
    return num_fixs_to_strip

