"""Compute target identifications from fixation data (stage 2).

Replaces the old data_models/preprocess/target_identifications.py, which operated on Trial objects
and raw 600 Hz gaze. This version operates on persisted DataFrames (fixations + icons + actions +
metadata), using the fixation that contains the identification timestamp to determine gaze position.
See _determine_identification_source.ipynb for the feasibility analysis (98% during fixation, 99.9%
target agreement).
"""

from typing import Union, Sequence

import numpy as np
import pandas as pd

import constants as cnst
from data_models.LWSEnums import SubjectActionCategoryEnum, SignalDetectionCategoryEnum
from pipeline.utils import pixel_distance


def build_identifications(
        fixations: pd.DataFrame,
        icons: pd.DataFrame,
        actions: pd.DataFrame,
        metadata: pd.DataFrame,
        identification_actions: Union[Sequence[SubjectActionCategoryEnum], SubjectActionCategoryEnum],
        on_target_threshold_dva: float,
) -> pd.DataFrame:
    """Build the target-identification table from stage-1 outputs.

    :param fixations: the fixation subset of eye_movements (dominant eye only).
    :param icons: full icon table with is_target.
    :param actions: subject actions table.
    :param metadata: per (subject, trial) metadata with px2deg and duration.
    :param identification_actions: action(s) that count as an identification attempt.
    :param on_target_threshold_dva: distance threshold for a hit.

    :return: DataFrame with columns: subject, trial, target, identification_category, time,
        to_trial_end, distance_px, distance_dva.
    """
    if isinstance(identification_actions, SubjectActionCategoryEnum):
        identification_actions = [identification_actions]
    identification_actions = list(set(identification_actions))

    targets = icons.loc[icons["is_target"]].rename(columns={cnst.ICON_STR: cnst.TARGET_STR})

    px2deg_lookup = (
        metadata
        .drop_duplicates(subset=[cnst.SUBJECT_STR])
        .set_index(cnst.SUBJECT_STR)["px2deg"]
    )

    trial_end_lookup = (
        metadata
        .set_index([cnst.SUBJECT_STR, cnst.TRIAL_STR])["duration"]
    )

    results = []
    for (subj, trial), trial_actions in actions.groupby(
        [cnst.SUBJECT_STR, cnst.TRIAL_STR], sort=False, observed=True,
    ):
        ident_actions = trial_actions.loc[
            trial_actions[cnst.ACTION_STR].isin(identification_actions)
        ]
        if ident_actions.empty:
            trial_targets = targets.loc[
                (targets[cnst.SUBJECT_STR] == subj) & (targets[cnst.TRIAL_STR] == trial)
            ]
            misses = _make_misses(trial_targets[cnst.TARGET_STR].to_numpy(), subj, trial)
            if not misses.empty:
                results.append(misses)
            continue

        trial_fixations = fixations.loc[
            (fixations[cnst.SUBJECT_STR] == subj) & (fixations[cnst.TRIAL_STR] == trial)
        ]
        trial_targets = targets.loc[
            (targets[cnst.SUBJECT_STR] == subj) & (targets[cnst.TRIAL_STR] == trial)
        ]
        if trial_targets.empty:
            continue

        conv = px2deg_lookup.loc[subj]
        trial_end = trial_end_lookup.loc[(subj, trial)]

        ident_times = ident_actions[cnst.TIME_STR].to_numpy()
        tgt_x = trial_targets[cnst.X].to_numpy()
        tgt_y = trial_targets[cnst.Y].to_numpy()
        tgt_ids = trial_targets[cnst.TARGET_STR].to_numpy()

        ident_rows = []
        for t in ident_times:
            fix_x, fix_y = _gaze_at_time(t, trial_fixations)
            if np.isnan(fix_x):
                continue
            dists_px = pixel_distance(fix_x, fix_y, tgt_x, tgt_y)
            nearest_idx = int(np.argmin(dists_px))
            ident_rows.append({
                cnst.SUBJECT_STR: subj,
                cnst.TRIAL_STR: trial,
                cnst.TARGET_STR: tgt_ids[nearest_idx],
                cnst.TIME_STR: t,
                "to_trial_end": trial_end - t,
                cnst.DISTANCE_PX_STR: dists_px[nearest_idx],
                cnst.DISTANCE_DVA_STR: dists_px[nearest_idx] * conv,
            })

        if not ident_rows:
            misses = _make_misses(tgt_ids, subj, trial)
            if not misses.empty:
                results.append(misses)
            continue

        idents = pd.DataFrame(ident_rows)
        idents = _classify_hits_and_false_alarms(idents, on_target_threshold_dva)
        idents = _append_missed_targets(idents, tgt_ids)
        results.append(idents)

    if not results:
        return _empty_result()
    return pd.concat(results, ignore_index=True)


def _gaze_at_time(t: float, trial_fixations: pd.DataFrame) -> tuple[float, float]:
    """Find the gaze position at time t from the fixation table.

    Returns the (x, y) of the fixation that contains t, or the most recent fixation
    that ended within 50 ms before t. Returns (NaN, NaN) if no suitable fixation is found.
    """
    JUST_AFTER_MS = 50.0
    during = trial_fixations.loc[
        (trial_fixations[cnst.START_TIME_STR] <= t) & (trial_fixations[cnst.END_TIME_STR] >= t)
    ]
    if not during.empty:
        row = during.iloc[0]
        return row[cnst.X], row[cnst.Y]
    before = trial_fixations.loc[
        (trial_fixations[cnst.END_TIME_STR] < t) & (t - trial_fixations[cnst.END_TIME_STR] <= JUST_AFTER_MS)
    ]
    if not before.empty:
        row = before.iloc[-1]
        return row[cnst.X], row[cnst.Y]
    return np.nan, np.nan


def _classify_hits_and_false_alarms(idents: pd.DataFrame, on_target_threshold_dva: float) -> pd.DataFrame:
    idents = idents.copy().sort_values(by=[cnst.TIME_STR, cnst.TARGET_STR])
    idents[cnst.IDENTIFICATION_CATEGORY_STR] = idents[cnst.DISTANCE_DVA_STR].map(
        lambda dist: SignalDetectionCategoryEnum.HIT if dist <= on_target_threshold_dva
        else SignalDetectionCategoryEnum.FALSE_ALARM
    )
    is_hit = idents[cnst.IDENTIFICATION_CATEGORY_STR] == SignalDetectionCategoryEnum.HIT
    is_repeated_hit = idents.loc[is_hit, cnst.TARGET_STR].duplicated(keep="first")
    idents.loc[is_hit & is_repeated_hit, cnst.IDENTIFICATION_CATEGORY_STR] = SignalDetectionCategoryEnum.REPEATED_HIT
    is_false_alarm = idents[cnst.IDENTIFICATION_CATEGORY_STR] == SignalDetectionCategoryEnum.FALSE_ALARM
    idents.loc[is_false_alarm, cnst.TARGET_STR] = None
    return idents


def _append_missed_targets(idents: pd.DataFrame, target_ids: np.ndarray) -> pd.DataFrame:
    hit_targets = idents.loc[
        idents[cnst.IDENTIFICATION_CATEGORY_STR] == SignalDetectionCategoryEnum.HIT, cnst.TARGET_STR
    ]
    missed = target_ids[~np.isin(target_ids, hit_targets.unique())]
    if len(missed) == 0:
        return idents
    subj = idents[cnst.SUBJECT_STR].iloc[0]
    trial = idents[cnst.TRIAL_STR].iloc[0]
    misses = _make_misses(missed, subj, trial)
    return pd.concat([idents, misses], ignore_index=True)


def _make_misses(target_ids: np.ndarray, subject: int, trial: int) -> pd.DataFrame:
    if len(target_ids) == 0:
        return pd.DataFrame()
    return pd.DataFrame({
        cnst.SUBJECT_STR: subject,
        cnst.TRIAL_STR: trial,
        cnst.TARGET_STR: target_ids,
        cnst.IDENTIFICATION_CATEGORY_STR: SignalDetectionCategoryEnum.MISS,
        cnst.TIME_STR: np.inf,
        "to_trial_end": np.nan,
        cnst.DISTANCE_PX_STR: np.inf,
        cnst.DISTANCE_DVA_STR: np.inf,
    })


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        cnst.SUBJECT_STR, cnst.TRIAL_STR, cnst.TARGET_STR,
        cnst.IDENTIFICATION_CATEGORY_STR, cnst.TIME_STR,
        "to_trial_end", cnst.DISTANCE_PX_STR, cnst.DISTANCE_DVA_STR,
    ])
