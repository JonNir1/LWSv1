"""Compute fixation-to-target distances in long format (the T4 fix).

Replaces the wide per-target distance columns that used to live in `eye_movements.pkl`. Those columns
were unique per trial (different trials have different targets), so concatenating subjects produced a
97%-NaN table. This module computes the same distances on-the-fly from the persisted stage-1 outputs.
"""

import numpy as np
import pandas as pd

import constants as cnst
from pipeline.utils import pixel_distance


def fixations_to_targets(
        fixations: pd.DataFrame,
        icons: pd.DataFrame,
        metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Compute distances from every fixation to every target in its trial.

    :param fixations: the fixation subset of eye_movements (event_type == FIXATION).
    :param icons: the full icon table (all 180 icons per trial, with is_target).
    :param metadata: per (subject, trial) metadata; must contain a `px2deg` column.

    :return: long-format DataFrame with columns:
        subject, trial, eye, event, target, distance_px, distance_dva
    One row per (fixation, target) pair. Fixations with NaN x/y are excluded.
    """
    targets = icons.loc[icons["is_target"]].copy()
    targets = targets.rename(columns={cnst.ICON_STR: cnst.TARGET_STR})

    if fixations.empty or targets.empty:
        return _empty_result()
    valid_fixations = fixations.dropna(subset=[cnst.X, cnst.Y])
    if valid_fixations.empty:
        return _empty_result()

    px2deg_lookup = (
        metadata
        .drop_duplicates(subset=[cnst.SUBJECT_STR])
        .set_index(cnst.SUBJECT_STR)["px2deg"]
    )

    results = []
    for (subj, trial), fix_group in valid_fixations.groupby(
        [cnst.SUBJECT_STR, cnst.TRIAL_STR], sort=False, observed=True,
    ):
        trial_targets = targets.loc[
            (targets[cnst.SUBJECT_STR] == subj) & (targets[cnst.TRIAL_STR] == trial)
        ]
        if trial_targets.empty:
            continue

        fix_x = fix_group[cnst.X].to_numpy()[:, None]
        fix_y = fix_group[cnst.Y].to_numpy()[:, None]
        tgt_x = trial_targets[cnst.X].to_numpy()[None, :]
        tgt_y = trial_targets[cnst.Y].to_numpy()[None, :]
        dists_px = pixel_distance(fix_x, fix_y, tgt_x, tgt_y)

        conv = px2deg_lookup.loc[subj]
        dists_dva = dists_px * conv

        tgt_ids = trial_targets[cnst.TARGET_STR].to_numpy()
        n_fix = len(fix_group)
        n_tgt = len(trial_targets)

        chunk = pd.DataFrame({
            cnst.SUBJECT_STR: np.repeat(fix_group[cnst.SUBJECT_STR].to_numpy(), n_tgt),
            cnst.TRIAL_STR: np.repeat(fix_group[cnst.TRIAL_STR].to_numpy(), n_tgt),
            cnst.EYE_STR: np.repeat(fix_group[cnst.EYE_STR].to_numpy(), n_tgt),
            cnst.EVENT_STR: np.repeat(fix_group[cnst.EVENT_STR].to_numpy(), n_tgt),
            cnst.TARGET_STR: np.tile(tgt_ids, n_fix),
            f"{cnst.DISTANCE_STR}_px": dists_px.ravel(),
            cnst.DISTANCE_DVA_STR: dists_dva.ravel(),
        })
        results.append(chunk)

    if not results:
        return _empty_result()
    return pd.concat(results, ignore_index=True)


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        cnst.SUBJECT_STR, cnst.TRIAL_STR, cnst.EYE_STR, cnst.EVENT_STR,
        cnst.TARGET_STR, f"{cnst.DISTANCE_STR}_px", cnst.DISTANCE_DVA_STR,
    ])
