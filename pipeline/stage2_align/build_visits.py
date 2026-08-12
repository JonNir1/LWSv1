"""Group consecutive on-target fixations into visits.

A visit is an episode of looking at a target: a sequence of fixations from the same eye that are all
within the on-target distance threshold, with no temporal gap exceeding the merging threshold. Visits
exist only for targets; there is no general clustering of fixations.
"""

import numpy as np
import pandas as pd

import constants as cnst


def build_visits(
        fixations: pd.DataFrame,
        fixation_target_dists: pd.DataFrame,
        on_target_threshold_dva: float,
        visit_merging_time_threshold: float,
) -> pd.DataFrame:
    """Build the visits table from fixations and their target distances.

    :param fixations: fixation events (dominant eye, non-outlier).
    :param fixation_target_dists: long-format output of fixations_to_targets().
    :param on_target_threshold_dva: distance threshold for on-target.
    :param visit_merging_time_threshold: max gap (ms) between consecutive on-target fixations to merge.
    :return: one row per (subject, trial, eye, target, visit).
    """
    assert on_target_threshold_dva > 0, "on_target_threshold_dva must be positive."
    assert visit_merging_time_threshold > 0, "visit_merging_time_threshold must be positive."

    on_target = fixation_target_dists.loc[
        fixation_target_dists[cnst.DISTANCE_DVA_STR] <= on_target_threshold_dva
    ]
    if on_target.empty:
        return pd.DataFrame()

    fix_key = [cnst.SUBJECT_STR, cnst.TRIAL_STR, cnst.EYE_STR, cnst.EVENT_STR]
    fix_cols = [cnst.START_TIME_STR, cnst.END_TIME_STR, "duration", "to_trial_end",
                cnst.X, cnst.Y, "num_fixs_to_strip"]
    outlier_cols = ["outlier_reasons"] if "outlier_reasons" in fixations.columns else []
    fix_subset = fixations[fix_key + fix_cols + outlier_cols].copy()

    merged = on_target.merge(fix_subset, on=fix_key, how="left")

    visits = []
    for (subj, trial, eye, target), group in merged.groupby(
        [cnst.SUBJECT_STR, cnst.TRIAL_STR, cnst.EYE_STR, cnst.TARGET_STR],
        sort=False, observed=True,
    ):
        group = group.sort_values(cnst.EVENT_STR).reset_index(drop=True)
        visit_ids = _assign_visit_ids(group, visit_merging_time_threshold)
        for vid in visit_ids.dropna().unique():
            visit_fixs = group.loc[visit_ids == vid]
            visits.append(_extract_visit_features(
                visit_fixs, int(vid), subj, trial, target, eye,
            ))

    if not visits:
        return pd.DataFrame()
    return pd.DataFrame(visits)


def _assign_visit_ids(
        on_target_fixs: pd.DataFrame,
        visit_merging_time_threshold: float,
) -> pd.Series:
    """Assign visit IDs to consecutive on-target fixations for a single (trial, eye, target)."""
    time_diffs = on_target_fixs[cnst.START_TIME_STR] - on_target_fixs[cnst.END_TIME_STR].shift(1)
    time_diffs = time_diffs.fillna(np.inf)
    is_new_visit = time_diffs > visit_merging_time_threshold
    return is_new_visit.cumsum().astype(float)


def _extract_visit_features(
        visit_fixs: pd.DataFrame, visit_idx: int,
        subject: int, trial: int, target: str, eye: str,
) -> dict:
    durations = visit_fixs["duration"]
    total_dur = np.nansum(durations)
    center_x = np.nansum(visit_fixs[cnst.X].to_numpy() * durations.to_numpy()) / total_dur
    center_y = np.nansum(visit_fixs[cnst.Y].to_numpy() * durations.to_numpy()) / total_dur
    dists_dva = visit_fixs[cnst.DISTANCE_DVA_STR]
    weighted_dist = np.nansum(dists_dva.to_numpy() * durations.to_numpy()) / total_dur

    num_outliers = 0
    if "outlier_reasons" in visit_fixs.columns:
        is_outlier = visit_fixs["outlier_reasons"].map(lambda r: isinstance(r, list) and len(r) > 0)
        num_outliers = int(is_outlier.sum())

    return {
        cnst.SUBJECT_STR: subject,
        cnst.TRIAL_STR: trial,
        cnst.EYE_STR: eye,
        cnst.TARGET_STR: target,
        cnst.VISIT_STR: visit_idx,
        cnst.EVENT_STR: sorted(visit_fixs[cnst.EVENT_STR].to_numpy()),
        cnst.START_TIME_STR: visit_fixs[cnst.START_TIME_STR].iloc[0],
        cnst.END_TIME_STR: visit_fixs[cnst.END_TIME_STR].iloc[-1],
        "duration": visit_fixs[cnst.END_TIME_STR].iloc[-1] - visit_fixs[cnst.START_TIME_STR].iloc[0],
        "to_trial_end": visit_fixs["to_trial_end"].iloc[-1],
        cnst.X: center_x,
        cnst.Y: center_y,
        f"min_{cnst.DISTANCE_STR}_dva": dists_dva.min(),
        f"max_{cnst.DISTANCE_STR}_dva": dists_dva.max(),
        f"weighted_{cnst.DISTANCE_STR}_dva": weighted_dist,
        "num_fixs_to_strip": visit_fixs["num_fixs_to_strip"].iloc[-1],
        "num_fixations": len(visit_fixs),
        "num_outlier_fixations": num_outliers,
    }
