import os

import numpy as np
import pandas as pd

import config as cnfg
from data_models.LWSEnums import TargetVisitTypeEnum
from data_models.Subject import Subject


def get_visits(
        subject: Subject,
        distance_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
        time_window_ms: float = cnfg.CHUNKING_TEMPORAL_WINDOW_MS,
        save: bool = True,
        verbose: bool = False
) -> pd.DataFrame:
    """
    Get the visits DataFrame for a subject.
    If the DataFrame is not found in the subject's output directory, it will be generated from the subject's fixations.

    :param subject: Subject object containing trial data.
    :param distance_threshold_dva: float - the distance threshold in degrees of visual angle (DVA) to consider a fixation as "on target".
    :param time_window_ms: float - the temporal window in milliseconds to consider fixations as part of the same visit.
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
        from analysis.fixations import get_fixations
        fixations = get_fixations(subject, save=False, verbose=False)
        fixations = fixations[fixations["outlier_reasons"].apply(lambda x: len(x) == 0)]  # drop outliers
        visits_df = extract_visits(
            fixations,
            subject.get_target_identification_summary(verbose=False),
            1.0 / subject.px2deg,
            distance_threshold_dva,
            time_window_ms,
        )
        if save:
            visits_df.to_pickle(path)
        return visits_df


def extract_visits(
        fixations_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        deg2px_coeff: float,
        distance_threshold_dva: float,
        temporal_window_ms: float,
) -> pd.DataFrame:
    """
    Extracts visits from the given fixations DataFrame, merging it with the targets DataFrame to include target
    information. A visit is defined as a sequence of consecutive fixations that are within a specified distance
    threshold (in degrees of visual angle) from the same target. A *new* visit is initiated when one of two conditions is met:
    - the current fixation is on the target and the previous was not,
    - OR both are on-target, but their time gap exceeds the specified temporal window.

    The resulting DataFrame contains the following columns:
    - trial: int - the trial number
    - eye: str - the eye (e.g., "left", "right")
    - target: str - the target name (e.g., "target_1", "target_2")
    - visit: int - the visit index
    - start_fixation, end_fixation: int - the ID of the first and last fixations in the visit
    - num_fixations: int - the number of fixations in the visit
    - start_time, end_time: float - the start and end time of the visit (in ms)
    - duration: float - the duration of the visit (in ms)
    - to_trial_end: float - the time from the visit's end to the trial's end (in ms)
    - visit_x, visit_y: float - the average pixel coordinates of the visit's fixations
    - min_distance, max_distance: float - the distance of the visit's closest and farthest fixations to the target (in DVA)
    - mean_distance: float - the mean distance of the visit's fixations to the target (in pixels)
    - weighted_distance: float - the average distance, weighted by fixation durations (in DVA)
    - next_1_in_strip, next_2_in_strip, next_3_in_strip: bool; whether the next 1, 2, or 3 visits are in the bottom strip
    - target_time: float - the time of the target identification in the trial, or inf if the target was never identified
    - target_x, target_y, target_angle: float - the pixel coordinates and angle of the target
    - target_category: int - the category of the target (see ImageCategoryEnum)
    - trial_type: int - the type of the trial (e.g., "color", "bw", "noise"; see SearchArrayTypeEnum)
    - trial_duration: float - the duration of the trial (in ms)
    - bad_trial: bool - whether the trial is considered "bad" (e.g., if the subject performed a "bad" action)
    """
    visits_df = _convert_fixations_to_visits(fixations_df, deg2px_coeff, distance_threshold_dva, temporal_window_ms)
    visits_df = visits_df.set_index([cnfg.TRIAL_STR, cnfg.TARGET_STR])
    targets_df = (
        targets_df
        .set_index([cnfg.TRIAL_STR, cnfg.TARGET_STR])
        .rename(columns={cnfg.TIME_STR: cnfg.TARGET_TIME_STR})
        .drop(columns=[col for col in targets_df.columns if col.startswith(cnfg.DISTANCE_STR)], errors='ignore')
        .loc[visits_df.index]
    )
    mean_dist = _visit_to_target_distance(visits_df, targets_df)
    visit_type = _classify_visit_type(visits_df, targets_df)
    merged = (
        pd.concat([visits_df, targets_df, mean_dist, visit_type], axis=1)
        .reset_index(drop=False)
        .sort_values([cnfg.TRIAL_STR, cnfg.TARGET_STR, cnfg.EYE_STR, cnfg.VISIT_STR], inplace=False)
        .reset_index(drop=True)
    )
    return merged


def _convert_fixations_to_visits(
        fixations_df: pd.DataFrame,
        deg2px_coeff: float,
        distance_threshold_dva: float,
        temporal_window_ms: float,
) -> pd.DataFrame:
    """
    Converts the fixations DataFrame into a `visits` DataFrame for a given subject. A `visit` is defined as a sequence
    of consecutive fixations that are within a specified distance threshold (in degrees of visual angle) from the same
    target.

    The DataFrame contains the following columns:
    - trial: int - the trial number
    - eye: str - the eye (e.g., "left", "right")
    - target: str - the target name (e.g., "target_1", "target_2")
    - visit: int - the visit index
    - start_fixation, end_fixation: int - the ID of the first and last fixations in the visit
    - num_fixations: int - the number of fixations in the visit
    - start_time, end_time: float - the start and end time of the visit (in ms)
    - duration: float - the duration of the visit (in ms)
    - to_trial_end: float - the time from the visit's end to the trial's end (in ms)
    - visit_x, visit_y: float - the average pixel coordinates of the visit's fixations
    - min_distance, max_distance: float - the distance of the visit's closest and farthest fixations to the target (in DVA)
    - weighted_distance: float - the average distance, weighted by fixation durations (in DVA)
    - next_1_in_strip, next_2_in_strip, next_3_in_strip: bool; whether the next 1, 2, or 3 visits are in the bottom strip
    """
    assert deg2px_coeff > 0, "`deg2px_coeff` must be positive."
    assert distance_threshold_dva > 0, "`distance_threshold_dva` must be positive."
    assert temporal_window_ms > 0, "`temporal_window_ms` must be positive."
    visits = []
    for (trial, eye), group in fixations_df.groupby(by=[cnfg.TRIAL_STR, cnfg.EYE_STR]):
        group = group.sort_values(cnfg.FIXATION_STR, inplace=False).reset_index(drop=True)
        for target in group[cnfg.TARGET_STR].unique():
            visit_ids = _assign_visit_ids(group, distance_threshold_dva, temporal_window_ms)
            if visit_ids.isna().all():
                continue
            for visit_idx in visit_ids.dropna().unique():
                visit_fixs = group[visit_ids == visit_idx]
                visits.append(
                    _extract_features_from_fixs(visit_fixs, visit_idx, trial, target, eye)
                )
    return pd.DataFrame(visits)


def _assign_visit_ids(
        visit_fixs: pd.DataFrame,
        distance_threshold_dva: float,
        temporal_window_ms: float,
) -> pd.Series:
    """
    Assigns visit IDs to fixations if they are part of a target-visit, or NaN otherwise.
    A *new* visit is started when one of two conditions is met:
    - the current fixation is on the target and the previous was not,
    - OR both are on-target, bet their time gap exceeds the temporal window.

    :param visit_fixs: DataFrame containing fixation data, with columns for target distances and start and end times.
    :param distance_threshold_dva: Distance in DVA to consider a fixation as "on target".
    :param temporal_window_ms: Time window in milliseconds to consider two fixations as part of the same visit.
    :return: A Series with visit IDs, indexed by the same index as `fix_group`.
    """
    assert distance_threshold_dva > 0, "Distance threshold must be positive."
    assert temporal_window_ms > 0, "Temporal threshold must be positive."
    on_target = visit_fixs[cnfg.DISTANCE_STR] <= distance_threshold_dva
    prev_on_target = on_target.shift(1, fill_value=False)  # consider the 1st fixation's previous as not on target
    time_diffs = visit_fixs["start_time"] - visit_fixs["end_time"].shift(1)
    outside_time_window = (time_diffs > temporal_window_ms).fillna(
        True)  # consider the 1st fixation's previous as outside the time window
    is_new_visit = on_target & (~prev_on_target | outside_time_window)
    visit_counter = is_new_visit.cumsum()
    visit_id = np.where(on_target, visit_counter, np.nan)
    visit_id = pd.Series(visit_id, index=visit_fixs.index)
    return visit_id


def _extract_features_from_fixs(visit_fixs: pd.DataFrame, visit_idx: int, trial: int, target: str, eye: str,) -> dict:
    """ Extracts a visit's features from its underlying fixations. """
    assert not visit_fixs.empty, f"Visit {visit_idx} for trial {trial}, target {target}, eye {eye} is empty."
    durations = visit_fixs["duration"]
    center_pixel_x = np.nansum(visit_fixs[cnfg.X].values * durations.values) / np.nansum(durations)
    center_pixel_y = np.nansum(visit_fixs[cnfg.Y].values * durations.values) / np.nansum(durations)
    weighted_distance = np.nansum(visit_fixs[cnfg.DISTANCE_STR].values * durations.values) / np.nansum(durations)
    return {
        cnfg.TRIAL_STR: trial,
        cnfg.EYE_STR: eye,
        cnfg.TARGET_STR: target,
        cnfg.VISIT_STR: int(visit_idx),
        f"start_{cnfg.FIXATION_STR}": int(visit_fixs[cnfg.FIXATION_STR].iloc[0]),
        f"end_{cnfg.FIXATION_STR}": int(visit_fixs[cnfg.FIXATION_STR].iloc[-1]),
        "num_fixations": len(visit_fixs),
        cnfg.START_TIME_STR: visit_fixs[cnfg.START_TIME_STR].iloc[0],
        cnfg.END_TIME_STR: visit_fixs[cnfg.END_TIME_STR].iloc[-1],
        "duration": visit_fixs[cnfg.END_TIME_STR].iloc[-1] - visit_fixs[cnfg.START_TIME_STR].iloc[0],
        f"to_trial_end": visit_fixs["to_trial_end"].iloc[-1],
        f"{cnfg.VISIT_STR}_{cnfg.X}": center_pixel_x,
        f"{cnfg.VISIT_STR}_{cnfg.Y}": center_pixel_y,
        f"min_{cnfg.DISTANCE_STR}": visit_fixs[cnfg.DISTANCE_STR].min(),
        f"max_{cnfg.DISTANCE_STR}": visit_fixs[cnfg.DISTANCE_STR].max(),
        f"weighted_{cnfg.DISTANCE_STR}": weighted_distance,
        f"next_1_{cnfg.FIXATION_STR}_in_strip": visit_fixs["next_1_in_strip"].iloc[-1],
        f"next_2_{cnfg.FIXATION_STR}_in_strip": visit_fixs["next_2_in_strip"].iloc[-1],
        f"next_3_{cnfg.FIXATION_STR}_in_strip": visit_fixs["next_3_in_strip"].iloc[-1],
    }


def _visit_to_target_distance(visits_df: pd.DataFrame, targets_df: pd.DataFrame) -> pd.Series:
    """
    Calculates the Euclidean distance between the center of each visit and the center of the corresponding target (in pixels).
    Assumes both `visits_df` and `targets_df` have an identical multiindex with levels (trial, target).
    """
    assert visits_df.index.equals(targets_df.index)
    visit_centers = visits_df[[f"{cnfg.VISIT_STR}_{cnfg.X}", f"{cnfg.VISIT_STR}_{cnfg.Y}"]].rename(
        columns={f"{cnfg.VISIT_STR}_{cnfg.X}": cnfg.X, f"{cnfg.VISIT_STR}_{cnfg.Y}": cnfg.Y}
    )
    target_centers = targets_df[[f"{cnfg.TARGET_STR}_{cnfg.X}", f"{cnfg.TARGET_STR}_{cnfg.Y}"]].rename(
        columns={f"{cnfg.TARGET_STR}_{cnfg.X}": cnfg.X, f"{cnfg.TARGET_STR}_{cnfg.Y}": cnfg.Y}
    )
    diffs = visit_centers - target_centers
    distances = np.sqrt(diffs[cnfg.X] ** 2 + diffs[cnfg.Y] ** 2).rename(f"mean_{cnfg.DISTANCE_STR}")
    return distances


def _classify_visit_type(visits_df: pd.DataFrame, targets_df: pd.DataFrame) -> pd.Series:
    """
    Classifies each visit as "before", "marking", or "after" the target time.
    Assumes both `visits_df` and `targets_df` have an identical multiindex with levels (trial, target).
    """
    assert visits_df.index.equals(targets_df.index)
    visit_start, visit_end = visits_df[cnfg.START_TIME_STR], visits_df[cnfg.END_TIME_STR]
    assert (visit_start <= visit_end).all(), "Visit start times must be less than or equal to end times."
    target_time = targets_df[cnfg.TARGET_TIME_STR].fillna(np.inf)
    visit_type = pd.Series(TargetVisitTypeEnum.OTHER, index=visits_df.index, name=f"{cnfg.VISIT_STR}_type")
    visit_type.loc[visit_end < target_time] = TargetVisitTypeEnum.BEFORE
    visit_type.loc[visit_start > target_time] = TargetVisitTypeEnum.AFTER
    visit_type.loc[(visit_start <= target_time) & (target_time <= visit_end)] = TargetVisitTypeEnum.MARKING
    return visit_type
