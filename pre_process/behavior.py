import warnings
from typing import Union, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cnfg
import helpers as hlp
from data_models.Subject import Subject
from data_models.Trial import Trial
from data_models.LWSEnums import SubjectActionTypesEnum, TargetIdentificationTypeEnum


def extract_behavior(
        subject: Subject,
        identification_actions: Union[Sequence[SubjectActionTypesEnum], SubjectActionTypesEnum],
        temporal_matching_threshold: float,
        false_alarm_threshold_dva: float,
        verbose: bool = False,
) -> pd.DataFrame:
    """
    Extracts the target identification behavior of a subject across all trials.

    :param subject: Subject object
    :param identification_actions: action(s) that indicate the subject has identified a target.
    :param temporal_matching_threshold: temporal threshold (in ms) for matching gaze samples to identification actions.
    :param false_alarm_threshold_dva: spatial threshold (in DVA) for classifying a target identification as a false alarm or a hit.
    :param verbose: if True, displays a progress bar for the extraction process.

    :return: a DataFrame containing the target identification behavior for each trial, with the following columns:
    - trial: int; the trial number
    - target: str; the name of the closest target to the subject's gaze at the time of identification
    - ident_type: TargetIdentificationTypeEnum; the type of identification (HIT, MISS, or FALSE_ALARM)
    - time: float; the time of the identification action in ms (relative to trial onset)
    - distance_px: float; the distance between the subject's gaze and the closest target, in pixels
    - distance_dva: float; the distance between the subject's gaze and the closest target, in DVA
    - left_x, left_y, right_x, right_y: float; the x and y coordinates of the subject's left and right eye gaze at the time of identification
    - left_pupil, right_pupil: float; the pupil size of the subject's left and right eye at the time of identification
    """
    trial_behaviors = {
        trial.trial_num: extract_trial_behavior(
            trial,
            identification_actions=identification_actions,
            temporal_matching_threshold=temporal_matching_threshold,
            false_alarm_threshold_dva=false_alarm_threshold_dva,
        )
        for trial in tqdm(subject.get_trials(), desc=f"Target Identifications", disable=not verbose)
    }
    behaviors = pd.concat(trial_behaviors.values(), axis=0, keys=trial_behaviors.keys())
    behaviors = (
        behaviors
        .reset_index(drop=False)
        .drop(
            columns=["target_sub_path", "level_1", "left_label", "right_label",],
            inplace=False,
            errors='ignore'
        )
        .rename(columns={"level_0": cnfg.TRIAL_STR})
        .sort_values(by=[cnfg.TRIAL_STR, cnfg.TARGET_STR])
    )
    return behaviors


def extract_trial_behavior(
        trial: Trial,
        identification_actions: Union[Sequence[SubjectActionTypesEnum], SubjectActionTypesEnum],
        temporal_matching_threshold: float,
        false_alarm_threshold_dva: float,
):
    ident_times = _extract_identification_times(trial, identification_actions)
    ident_gaze = _extract_gaze_on_identification(
        trial, identification_times=ident_times, temporal_matching_threshold=temporal_matching_threshold,
    )
    ident_dists = _find_closest_target(identification_gaze=ident_gaze, px2deg=trial.px2deg, )
    idents = pd.concat([
        ident_times.astype(float),
        ident_dists,
        ident_gaze[[col for col in ident_gaze.columns if col.startswith("left")]],
        ident_gaze[[col for col in ident_gaze.columns if col.startswith("right")]]
    ], axis=1)
    idents = _classify_behavior(idents, trial.get_targets(), false_alarm_threshold_dva)
    idents = _clean_fa_data(idents)

    # reorder columns
    ordered_cols = [cnfg.TARGET_STR, "ident_type"]
    ordered_cols += [col for col in idents.columns if col not in ordered_cols]
    idents = idents[ordered_cols]

    # warn if a target was identified multiple times
    non_fa_targets = idents.loc[idents["ident_type"] != TargetIdentificationTypeEnum.FALSE_ALARM, cnfg.TARGET_STR]
    non_fa_target_counts = non_fa_targets.value_counts()
    if (non_fa_target_counts > 1).any():
        # TODO: consider resolving this in code rather than manually?
        multi_detected = non_fa_target_counts[non_fa_target_counts > 1].index.tolist()
        warnings.warn(
            f"Multiple identifications for targets {multi_detected} in trial {trial.trial_num}.",
            RuntimeWarning,
        )
    return idents


def _extract_identification_times(
        trial: Trial,
        identification_actions: Union[Sequence[SubjectActionTypesEnum], SubjectActionTypesEnum],
) -> pd.Series:
    """ Extracts the identification times from the subject's actions during the trial. """
    if isinstance(identification_actions, SubjectActionTypesEnum):
        identification_actions = [identification_actions]
    identification_actions = list(set(identification_actions))
    actions = trial.get_actions()
    identification_times = actions.loc[np.isin(actions[cnfg.ACTION_STR], identification_actions)]
    identification_times = identification_times.reset_index(drop=True)
    return identification_times[cnfg.TIME_STR]


def _extract_gaze_on_identification(
        trial: Trial,
        identification_times: pd.Series,
        temporal_matching_threshold: float,
) -> pd.DataFrame:
    """
    Finds the gaze samples that corresponds to the subject's identification actions in a trial, and returns them.
    The gaze samples must be within `temporal_matching_threshold` ms before/after identification.
    """
    gaze = trial.get_gaze()
    gaze_times_for_ident_times = hlp.closest_indices(
        gaze[cnfg.TIME_STR], identification_times, threshold=temporal_matching_threshold
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
    dists = identification_gaze[[col for col in identification_gaze.columns if col.startswith(cnfg.TARGET_STR)]].copy()
    closest_target = dists.idxmin(axis=1).rename(cnfg.TARGET_STR)
    dists_px = pd.Series(
        dists.to_numpy()[dists.index, dists.columns.get_indexer(closest_target)],
        name=f"{cnfg.DISTANCE_STR}_px",
    )
    dists_dva = (dists_px * px2deg).rename(f"{cnfg.DISTANCE_STR}_dva")
    dists = pd.concat([closest_target, dists_px, dists_dva], axis=1)
    return dists


def _classify_behavior(
        identifications: pd.DataFrame, targets: pd.DataFrame, fa_threshold_dva: float,
) -> pd.DataFrame:
    assert fa_threshold_dva > 0, "False alarm threshold must be greater than 0."
    identifications["ident_type"] = identifications[f"{cnfg.DISTANCE_STR}_dva"].map(
        lambda dist: TargetIdentificationTypeEnum.FALSE_ALARM if dist > fa_threshold_dva else TargetIdentificationTypeEnum.HIT
    )

    # classify unidentified targets as misses
    all_targets = targets.index
    hit_targets = identifications.loc[
        identifications["ident_type"] == TargetIdentificationTypeEnum.HIT, cnfg.TARGET_STR
    ].unique()
    missed_targets = all_targets[np.isin(all_targets, hit_targets, invert=True)]
    misses = pd.DataFrame(index=range(len(missed_targets)))
    misses[cnfg.TARGET_STR] = missed_targets
    misses["ident_type"] = TargetIdentificationTypeEnum.MISS
    misses[[cnfg.TIME_STR, f"{cnfg.DISTANCE_STR}_px", f"{cnfg.DISTANCE_STR}_dva"]] = np.inf  # set unidentified times & dists to inf

    # append misses to the identifications DataFrame
    if not misses.empty:
        identifications = pd.concat([identifications, misses], axis=0)
    identifications = identifications.reset_index(drop=True, inplace=False)
    return identifications


def _clean_fa_data(idents_with_fa: pd.DataFrame) -> pd.DataFrame:
    """ Set false-alarm targets and target-distances to NaN """
    is_fa = idents_with_fa["ident_type"] == TargetIdentificationTypeEnum.FALSE_ALARM
    cleaned_cols = [cnfg.TARGET_STR] + [col for col in idents_with_fa.columns if cnfg.DISTANCE_STR in col]
    cleaned_idents = idents_with_fa.copy()
    cleaned_idents.loc[is_fa, cleaned_cols] = np.nan
    return cleaned_idents

