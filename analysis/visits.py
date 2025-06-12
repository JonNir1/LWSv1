import numpy as np
import pandas as pd

import config as cnfg
from data_models.LWSEnums import TargetVisitTypeEnum


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
    - min_distance, max_distance: float - the distance of the visit's closest and farthest fixations to the target (in pixels)
    - mean_distance: float - the mean distance of the visit's fixations to the target (in pixels)
    - next_1_fixation_in_strip, next_2_fixation_in_strip, next_3_fixation_in_strip: bool - whether the next 1st, 2nd,
        and 3rd fixations are in the bottom strip
    - target_time: float - the time of the target identification in the trial, or inf if the target was never identified
    - target_x, target_y, target_angle: float - the pixel coordinates and angle of the target
    - target_category: int - the category of the target (see ImageCategoryEnum)
    - identified: bool - whether the target was ever identified in the trial
    - trial_type: int - the type of the trial (e.g., "color", "bw", "noise"; see SearchArrayTypeEnum)
    - bad_trial: bool - whether the trial is considered "bad" (e.g., if the subject performed a "bad" action)
    """
    visits_df = _convert_fixations_to_visits(fixations_df, deg2px_coeff, distance_threshold_dva, temporal_window_ms)
    visits_df = visits_df.set_index([cnfg.TRIAL_STR, cnfg.TARGET_STR])
    targets_df = targets_df.set_index([cnfg.TRIAL_STR, cnfg.TARGET_STR]).rename(
        columns={cnfg.TIME_STR: f"{cnfg.TARGET_STR}_{cnfg.TIME_STR}"}
    ).loc[visits_df.index]
    mean_dist = _visit_to_target_distance(visits_df, targets_df)
    visit_type = _classify_visit_type(visits_df, targets_df)
    merged = pd.concat([visits_df, targets_df, mean_dist, visit_type], axis=1).reset_index(drop=False)
    merged = merged.set_index([cnfg.TRIAL_STR, cnfg.TARGET_STR, cnfg.EYE_STR, cnfg.VISIT_STR]).sort_index()
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
    - min_distance, max_distance: float - the distance of the visit's closest and farthest fixations to the target (in pixels)
    - next_1_fixation_in_strip, next_2_fixation_in_strip, next_3_fixation_in_strip: bool - whether the next 1st, 2nd,
        and 3rd fixations are in the bottom strip
    """
    assert deg2px_coeff > 0, "`deg2px_coeff` must be positive."
    assert distance_threshold_dva > 0, "`distance_threshold_dva` must be positive."
    assert temporal_window_ms > 0, "`temporal_window_ms` must be positive."
    distance_threshold_px = distance_threshold_dva * deg2px_coeff
    visits = []
    target_cols = [col for col in fixations_df.columns if col.startswith(f"{cnfg.TARGET_STR}_")]
    for (trial, eye), group in fixations_df.groupby(level=[cnfg.TRIAL_STR, cnfg.EYE_STR]):
        group = group.reset_index().sort_values(cnfg.FIXATION_STR, inplace=False)
        for target in target_cols:
            visit_ids = _assign_visit_ids(group, target, distance_threshold_px, temporal_window_ms)
            if visit_ids.isna().all():
                continue
            for visit_idx in visit_ids.dropna().unique():
                visit_fixs = group[visit_ids == visit_idx]
                visits.append(
                    _extract_features_from_fixs(visit_fixs, visit_idx, trial, target, eye)
                )
    visits_df = pd.DataFrame(visits)
    return visits_df


def _assign_visit_ids(
        visit_fixs: pd.DataFrame,
        target_name: str,
        distance_threshold_px: float,
        temporal_window_ms: float,
) -> pd.Series:
    """
    Assigns visit IDs to fixations if they are part of a target-visit, or NaN otherwise.
    A *new* visit is started when one of two conditions is met:
    - the current fixation is on the target and the previous was not,
    - OR both are on-target, bet their time gap exceeds the temporal window.

    :param visit_fixs: DataFrame containing fixation data, with columns for target distances and start and end times.
    :param target_name: Name of the column containing target distances.
    :param distance_threshold_px: Distance in pixels to consider a fixation as "on target".
    :param temporal_window_ms: Time window in milliseconds to consider two fixations as part of the same visit.
    :return: A Series with visit IDs, indexed by the same index as `fix_group`.
    """
    assert distance_threshold_px > 0, "Distance threshold must be positive."
    assert temporal_window_ms > 0, "Temporal threshold must be positive."
    on_target = visit_fixs[target_name] <= distance_threshold_px
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
    center_pixels = pd.DataFrame(visit_fixs["center_pixel"].to_list(), columns=[cnfg.X, cnfg.Y])
    durations = visit_fixs["duration"]
    center_pixel_x = np.nansum(center_pixels[cnfg.X].values * durations.values) / np.nansum(durations)
    center_pixel_y = np.nansum(center_pixels[cnfg.Y].values * durations.values) / np.nansum(durations)
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
        f"min_{cnfg.DISTANCE_STR}": visit_fixs[target].min(),
        f"max_{cnfg.DISTANCE_STR}": visit_fixs[target].max(),
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
    target_time = targets_df[f"{cnfg.TARGET_STR}_{cnfg.TIME_STR}"].fillna(np.inf)
    visit_type = pd.Series(TargetVisitTypeEnum.OTHER, index=visits_df.index, name=f"{cnfg.TARGET_STR}_type")
    visit_type.loc[visit_end < target_time] = TargetVisitTypeEnum.BEFORE
    visit_type.loc[visit_start > target_time] = TargetVisitTypeEnum.AFTER
    visit_type.loc[(visit_start <= target_time) & (target_time <= visit_end)] = TargetVisitTypeEnum.MARKING
    return visit_type
