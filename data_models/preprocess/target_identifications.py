from typing import Union, Sequence

import numpy as np
import pandas as pd

import constants as cnst
from data_models.Trial import Trial
from data_models.LWSEnums import SubjectActionCategoryEnum, SignalDetectionCategoryEnum


def extract_trial_identifications(
        trial: Trial,
        identification_actions: Union[Sequence[SubjectActionCategoryEnum], SubjectActionCategoryEnum],
        gaze_to_trigger_matching_threshold: float,
        on_target_threshold_dva: float,
):
    ident_times = _extract_identification_times(trial, identification_actions)
    ident_gaze = _extract_gaze_on_identification(
        trial,
        identification_times=ident_times[cnst.TIME_STR],
        gaze_to_trigger_matching_threshold=gaze_to_trigger_matching_threshold
    )
    ident_dists = _find_closest_target(identification_gaze=ident_gaze, px2deg=trial.px2deg, )
    idents = pd.concat([
        ident_times.astype(float),
        ident_dists,
        ident_gaze[[col for col in ident_gaze.columns if col.startswith("left")]],
        ident_gaze[[col for col in ident_gaze.columns if col.startswith("right")]]
    ], axis=1)

    # apply SDT classifications
    idents = _classify_hits_and_false_alarms(idents, on_target_threshold_dva)
    idents = _append_missed_targets(idents, trial.get_targets())

    # reorder columns
    ordered_cols = [cnst.TARGET_STR, cnst.IDENTIFICATION_CATEGORY_STR, cnst.TIME_STR]
    ordered_cols += [col for col in idents.columns if not (col in ordered_cols)]
    idents = idents[ordered_cols]
    return idents


def _extract_identification_times(
        trial: Trial,
        identification_actions: Union[Sequence[SubjectActionCategoryEnum], SubjectActionCategoryEnum],
) -> pd.DataFrame:
    """ Extracts the identification times and time-to-trial's-end from the subject's actions during the trial. """
    if isinstance(identification_actions, SubjectActionCategoryEnum):
        identification_actions = [identification_actions]
    identification_actions = list(set(identification_actions))
    actions = trial.get_actions()
    identification_times = (
        actions.loc[actions[cnst.ACTION_STR].isin(identification_actions), cnst.TIME_STR]
        .reset_index(drop=True)
        .rename(cnst.TIME_STR)
    )
    to_trial_end = (trial.end_time - identification_times).rename("to_trial_end")
    identification_times = pd.concat([identification_times, to_trial_end], axis=1)
    return identification_times


def _extract_gaze_on_identification(
        trial: Trial,
        identification_times: pd.Series,
        gaze_to_trigger_matching_threshold: float,
) -> pd.DataFrame:
    """
    Finds the gaze samples that corresponds to the subject's identification actions in a trial, and returns them.
    The gaze samples must be within `gaze_to_trigger_matching_threshold` ms before/after identification.
    """
    def closest_indices(s: pd.Series, vals: pd.Series, threshold: float) -> pd.Series:
        """ Finds indices in `s` whose values are closest to the values in `vals`, within a given threshold. """
        assert threshold >= 0, "Threshold must be non-negative"
        s_values = s.values
        val_values = vals.values

        # compute absolute differences between each val and all of s
        diffs = np.abs(s_values[None, :] - val_values[:, None], dtype=float)  # shape: (len(vals), len(s))
        diffs[diffs > threshold] = np.inf  # mask differences that exceed threshold
        closest_idx_in_s = diffs.argmin(axis=1)  # get index (in s) of the closest value for each val

        # Handle cases where all diffs were > threshold
        no_match = np.isinf(diffs.min(axis=1))
        result = pd.Series(s.index[closest_idx_in_s], index=vals.index)
        result[no_match] = np.nan
        return result

    gaze = trial.get_gaze()
    gaze_times_for_ident_times = closest_indices(
        gaze[cnst.TIME_STR], identification_times, threshold=gaze_to_trigger_matching_threshold
    )
    gaze_on_identification = gaze.loc[gaze_times_for_ident_times]
    gaze_on_identification = gaze_on_identification.reset_index(drop=True)
    return gaze_on_identification


def _find_closest_target(identification_gaze: pd.DataFrame, px2deg: float,) -> pd.DataFrame:
    """
    Finds the closest target to each gaze sample in the given DataFrame.
    :returns: A DataFrame assigning to each gaze sample the following:
    - The closest target's name
    - The distance to the closest target in pixels
    - The distance to the closest target in degrees of visual angle (DVA)
    """
    dists = identification_gaze[[col for col in identification_gaze.columns if col.startswith(cnst.TARGET_STR)]].copy()
    closest_target = dists.idxmin(axis=1).rename(cnst.TARGET_STR)
    dists_px = pd.Series(
        dists.to_numpy()[dists.index, dists.columns.get_indexer(closest_target)],
        name=cnst.DISTANCE_PX_STR,
    )
    dists_dva = (dists_px * px2deg).rename(cnst.DISTANCE_DVA_STR)
    dists = pd.concat([closest_target, dists_px, dists_dva], axis=1)
    return dists


def _classify_hits_and_false_alarms(idents: pd.DataFrame, on_target_threshold_dva: float) -> pd.DataFrame:
    """
    Classifies target-identifications as `hit` or `false_alarm` based on the distance from the target when subject
    was identifying it. Also Classifies `repeated_hit` as a target that was identified multiple times.
    Returns a copy of the original DataFrame with an additional column for the identification category.
    """
    assert on_target_threshold_dva > 0 and not np.isinf(on_target_threshold_dva), \
        f"On-target threshold must be positive and finite, got {on_target_threshold_dva}."
    if cnst.DISTANCE_DVA_STR not in idents.columns:
        raise KeyError(f"Column '{cnst.DISTANCE_DVA_STR}' not found in identifications DataFrame.")
    idents_copy = idents.copy().sort_values(by=[cnst.TIME_STR,  cnst.TARGET_STR])  # sort by time and target for consistency
    # classify hits and false alarms
    idents_copy[cnst.IDENTIFICATION_CATEGORY_STR] = idents_copy[cnst.DISTANCE_DVA_STR].map(
        lambda dist: SignalDetectionCategoryEnum.HIT if dist <= on_target_threshold_dva
        else SignalDetectionCategoryEnum.FALSE_ALARM
    )
    # classify repeated hits
    is_hit = idents_copy[cnst.IDENTIFICATION_CATEGORY_STR] == SignalDetectionCategoryEnum.HIT
    is_repeated_hit = idents_copy.loc[is_hit, cnst.TARGET_STR].duplicated(keep="first")
    idents_copy.loc[is_hit & is_repeated_hit, cnst.IDENTIFICATION_CATEGORY_STR] = SignalDetectionCategoryEnum.REPEATED_HIT
    return idents_copy


def _append_missed_targets(identifications: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    # find missed targets
    all_targets = targets.index
    hit_targets = identifications.loc[
        identifications[cnst.IDENTIFICATION_CATEGORY_STR] == SignalDetectionCategoryEnum.HIT, cnst.TARGET_STR
    ]
    missed_targets = all_targets[np.isin(all_targets, hit_targets.unique(), invert=True)]
    misses = pd.DataFrame(index=range(len(missed_targets)))
    misses[cnst.TARGET_STR] = missed_targets
    misses[cnst.IDENTIFICATION_CATEGORY_STR] = SignalDetectionCategoryEnum.MISS
    misses[[cnst.TIME_STR, cnst.DISTANCE_PX_STR, cnst.DISTANCE_DVA_STR]] = np.inf  # set unidentified times & dists to inf

    # append misses to the identifications DataFrame
    if not misses.empty:
        identifications = pd.concat([identifications, misses], axis=0)
    identifications = identifications.reset_index(drop=True, inplace=False)
    return identifications
