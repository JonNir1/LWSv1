"""Compute fixation-to-icon distances in long format (the T4 fix).

Replaces the wide per-target distance columns that used to live in `eye_movements.pkl`. Those columns
were unique per trial (different trials have different targets), so concatenating subjects produced a
97%-NaN table. This module computes the same distances on-the-fly from the persisted stage-1 outputs.

Generic over which icons are passed in: callers wanting target-only distances pre-filter
`icons.loc[icons["is_target"]]`; callers wanting array-wide coverage pass the full icon table.
"""

import numpy as np
import pandas as pd

import constants as cnst
from utils.distances import pixel_distance


def fixations_to_icons(
        fixations: pd.DataFrame,
        icons: pd.DataFrame,
        metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Compute distances from every fixation to every icon passed in.

    :param fixations: the fixation subset of eye_movements (event_type == FIXATION).
    :param icons: the icons to compute distances to (e.g. `icons.loc[icons["is_target"]]` for targets only,
        or the full icon table for array-wide coverage).
    :param metadata: per (subject, trial) metadata; must contain a `px2deg` column.

    :return: long-format DataFrame with columns:
        subject, trial, eye, event, icon, distance_px, distance_dva
    One row per (fixation, icon) pair. Fixations with NaN x/y are excluded.
    """
    icons = icons.copy()

    if fixations.empty or icons.empty:
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
        trial_icons = icons.loc[
            (icons[cnst.SUBJECT_STR] == subj) & (icons[cnst.TRIAL_STR] == trial)
        ]
        if trial_icons.empty:
            continue

        fix_x = fix_group[cnst.X].to_numpy()[:, None]
        fix_y = fix_group[cnst.Y].to_numpy()[:, None]
        icon_x = trial_icons[cnst.X].to_numpy()[None, :]
        icon_y = trial_icons[cnst.Y].to_numpy()[None, :]
        dists_px = pixel_distance(fix_x, fix_y, icon_x, icon_y)

        conv = px2deg_lookup.loc[subj]
        dists_dva = dists_px * conv

        icon_ids = trial_icons[cnst.ICON_STR].to_numpy()
        n_fix = len(fix_group)
        n_icons = len(trial_icons)

        chunk = pd.DataFrame({
            cnst.SUBJECT_STR: np.repeat(fix_group[cnst.SUBJECT_STR].to_numpy(), n_icons),
            cnst.TRIAL_STR: np.repeat(fix_group[cnst.TRIAL_STR].to_numpy(), n_icons),
            cnst.EYE_STR: np.repeat(fix_group[cnst.EYE_STR].to_numpy(), n_icons),
            cnst.EVENT_STR: np.repeat(fix_group[cnst.EVENT_STR].to_numpy(), n_icons),
            cnst.ICON_STR: np.tile(icon_ids, n_fix),
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
        cnst.ICON_STR, f"{cnst.DISTANCE_STR}_px", cnst.DISTANCE_DVA_STR,
    ])
