import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cnfg


def extract_visits(
        all_fixations: pd.DataFrame,
        target_distance_threshold_dva: float,
        visit_merging_time_threshold: float,
) -> pd.DataFrame:
    """
    Groups the provided fixations into discrete target-visits. A target-visit is a sequence of fixations from the same
    eye that are all on-target (i.e., below the distance threshold), and that the time gap between each fixation and the
    previous one is below the temporal threshold. So, a new target-visit is initiated when two conditions are met:
    (1) the current fixation is on-target;
    (2) A) the previous fixation was not on-target, OR
        B) the previous fixation ended more than the temporal threshold before the current fixation started.
    NOTE: A fixation may belong to a single visit per target, but could be considered a visit for multiple targets.

    :param all_fixations: pd.DataFrame; fixations to be grouped into visits.
    :param target_distance_threshold_dva: float; the distance threshold in DVA for a fixation to be considered on-target.
    :param visit_merging_time_threshold: float; the time threshold in ms for separating visits.
    :return: a DataFrame containing the visits, with the following columns:
    - trial: int; the trial number
    - eye: str; the eye (left or right) from which the visit fixations were recorded
    - target: str; the target name for which the visit was recorded
    - visit: int; the visit ID
    - event_id: list; the event IDs of the visit's underlying fixations
    - start_time: float; the start time of the visit in ms (relative to trial onset)
    - end_time: float; the end time of the visit in ms (relative to trial onset)
    - duration: float; the duration of the visit in ms
    - to_trial_end: float; the time from the end of the visit to the end of the trial in ms
    - x: float; the x coordinate of the visit's center in pixels (weighted by fixation durations)
    - y: float; the y coordinate of the visit's center in pixels (weighted by fixation durations)
    - min_distance_dva: float; the minimum distance from the visit's fixations to the target in DVA
    - max_distance_dva: float; the maximum distance from the visit's fixations to the target in DVA
    - weighted_distance_dva: float; the weighted average distance from the visit's fixations to the target in DVA (weighted by fixation durations)
    - num_fixs_to_strip: int; the number of fixations from the visit's last fixation until the next visit to the
        bottom-strip, or np.inf if no subsequent fixations were inside the bottom-strip.
    """
    visits = []
    for (trial, eye), subset in tqdm(
            all_fixations.groupby(by=[cnfg.TRIAL_STR, cnfg.EYE_STR]),
            desc="Extracting Visits", disable=True,
    ):
        subset = subset.sort_values("event_id", inplace=False).reset_index(drop=False)
        if subset.empty:
            continue
        visit_ids = _assign_visit_ids(subset, target_distance_threshold_dva, visit_merging_time_threshold)
        for target_visit_col in visit_ids.columns:
            tgt_vis_ids = visit_ids[target_visit_col]
            if tgt_vis_ids.isna().all():
                continue
            for vis_id in tgt_vis_ids.dropna().unique():
                visits.append(_extract_visit_features(
                    visit_fixs=subset[tgt_vis_ids == vis_id],
                    visit_idx=vis_id,
                    trial=trial,
                    target=target_visit_col.replace(f"_{cnfg.VISIT_STR}", ""),
                    eye=eye,
                ))
    return pd.DataFrame(visits)


def _assign_visit_ids(
        fixs_subset: pd.DataFrame,
        target_distance_threshold_dva: float,
        visit_merging_time_threshold: float,
) -> pd.DataFrame:
    """
    Assigns a numerical visit ID to fixations that share the same trial and eye.
    A target-visit is a sequence of fixations from the same eye that are all on-target (i.e., below the distance
    threshold), and that the time gap between each fixation and the previous one is below the temporal threshold.
    So, a new target-visit is initiated when two conditions are met:
    (1) the current fixation is on-target;
    (2) A) the previous fixation was not on-target, OR
        B) the previous fixation ended more than the temporal threshold before the current fixation started.
    NOTE: A fixation may belong to a single visit per target, but could be considered a visit for multiple targets.

    Returns a DataFrame with the same index as `fixs_subset`, containing visit IDs for each target.
    """
    assert target_distance_threshold_dva > 0, "Target distance threshold must be positive."
    assert visit_merging_time_threshold > 0, "Visit time threshold must be positive."
    trials = fixs_subset[cnfg.TRIAL_STR].unique()
    if len(trials) != 1:
        raise RuntimeError(f"Fixation subset is not from a single trial: {trials.tolist()}")
    eyes = fixs_subset[cnfg.EYE_STR].unique()
    if len(eyes) != 1:
        raise RuntimeError(
            f"Fixation subset from trial {trials.iloc[0]} is not from a single eye: {eyes.tolist()}"
        )
    # check temporal threshold
    time_diffs = fixs_subset["start_time"] - fixs_subset["end_time"].shift(1)
    time_diffs = time_diffs.fillna(np.inf)  # consider the 1st fixation's time-from-previous as infinity (allow visit-start)
    is_time_separated = (time_diffs > visit_merging_time_threshold).astype(bool)

    # assign visit IDs per target
    dist_dva_suffix = f"_{cnfg.DISTANCE_STR}_dva"
    dist_dva_cols = [col for col in fixs_subset.columns if col.endswith(dist_dva_suffix)]
    if len(dist_dva_cols) == 0:
        raise RuntimeError(
            f"Fixation subset from trial {trials.iloc[0]} and eye {eyes.iloc[0]} does not contain DVA distances."
        )
    target_visit_ids = []
    for col in dist_dva_cols:
        target_name = col.replace(dist_dva_suffix, "")
        target_dists = fixs_subset[col]
        if target_dists.isna().all():
            continue
        # check distance threshold
        is_on_target = target_dists <= target_distance_threshold_dva
        prev_on_target = is_on_target.shift(1, fill_value=False)  # consider the 1st fixation's previous as off-target
        # assign visit IDs
        is_new_visit = is_on_target & (~prev_on_target | is_time_separated)
        visit_counter = is_new_visit.cumsum()
        visit_id = np.where(is_on_target, visit_counter, np.nan)
        visit_id = pd.Series(visit_id, index=fixs_subset.index, name=f"{target_name}_{cnfg.VISIT_STR}")
        target_visit_ids.append(visit_id)
    visit_ids = pd.concat(target_visit_ids, axis=1)
    return visit_ids


def _extract_visit_features(visit_fixs: pd.DataFrame, visit_idx: int, trial: int, target: str, eye: str,) -> dict:
    """ Extracts a visit's features from its underlying fixations. """
    assert not visit_fixs.empty, f"Visit {visit_idx} for trial {trial}, target {target}, eye {eye} is empty."
    durations = visit_fixs["duration"]
    center_pixel_x = np.nansum(visit_fixs[cnfg.X].values * durations.values) / np.nansum(durations)
    center_pixel_y = np.nansum(visit_fixs[cnfg.Y].values * durations.values) / np.nansum(durations)
    distance_col = f"{target}_{cnfg.DISTANCE_STR}_dva"
    weighted_distance = np.nansum(visit_fixs[distance_col].values * durations.values) / np.nansum(durations)
    return {
        cnfg.TRIAL_STR: trial,
        cnfg.EYE_STR: eye,
        cnfg.TARGET_STR: target,
        cnfg.VISIT_STR: int(visit_idx),
        "event_id": sorted(visit_fixs["event_id"].values),
        cnfg.START_TIME_STR: visit_fixs[cnfg.START_TIME_STR].iloc[0],
        cnfg.END_TIME_STR: visit_fixs[cnfg.END_TIME_STR].iloc[-1],
        "duration": visit_fixs[cnfg.END_TIME_STR].iloc[-1] - visit_fixs[cnfg.START_TIME_STR].iloc[0],
        f"to_trial_end": visit_fixs["to_trial_end"].iloc[-1],
        cnfg.X: center_pixel_x,
        cnfg.Y: center_pixel_y,
        f"min_{cnfg.DISTANCE_STR}_dva": visit_fixs[distance_col].min(),
        f"max_{cnfg.DISTANCE_STR}_dva": visit_fixs[distance_col].max(),
        f"weighted_{cnfg.DISTANCE_STR}_dva": weighted_distance,
        "num_fixs_to_strip": visit_fixs["num_fixs_to_strip"].iloc[-1],
    }


