from typing import Union, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cnfg
import helpers as hlp
from data_models.Subject import Subject
from data_models.Trial import Trial
from data_models.LWSEnums import SubjectActionTypesEnum


def extract_behavior(
        subject: Subject,
        identification_actions: Union[Sequence[SubjectActionTypesEnum], SubjectActionTypesEnum],
        temporal_matching_threshold: float,
        verbose: bool = False,
) -> pd.DataFrame:
    """
    Extracts the target identification behavior of a subject across all trials.

    :param subject: Subject object
    :param identification_actions: action(s) that indicate the subject has identified a target.
    :param temporal_matching_threshold: temporal threshold (in ms) for matching gaze samples to identification actions.
    :param verbose: if True, displays a progress bar for the extraction process.

    :return: a DataFrame containing the target identification behavior for each trial, with the following columns:
    - trial: int; the trial number
    - target: str; the name of the closest target to the subject's gaze at the time of identification
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
        .reset_index(drop=True)
    )
    return behaviors


def extract_trial_behavior(
        trial: Trial,
        identification_actions: Union[Sequence[SubjectActionTypesEnum], SubjectActionTypesEnum],
        temporal_matching_threshold: float,
):
    ident_times = _extract_identification_times(trial, identification_actions)
    ident_gaze = _extract_gaze_on_identification(
        trial, identification_times=ident_times[cnfg.TIME_STR], temporal_matching_threshold=temporal_matching_threshold,
    )
    ident_dists = _find_closest_target(identification_gaze=ident_gaze, px2deg=trial.px2deg, )
    idents = pd.concat([
        ident_times.astype(float),
        ident_dists,
        ident_gaze[[col for col in ident_gaze.columns if col.startswith("left")]],
        ident_gaze[[col for col in ident_gaze.columns if col.startswith("right")]]
    ], axis=1)
    idents = _classify_behavior(idents, trial.get_targets())

    # reorder columns
    ordered_cols = [cnfg.TARGET_STR] + [col for col in idents.columns if col != cnfg.TARGET_STR]
    idents = idents[ordered_cols]
    return idents


def _extract_identification_times(
        trial: Trial,
        identification_actions: Union[Sequence[SubjectActionTypesEnum], SubjectActionTypesEnum],
) -> pd.DataFrame:
    """ Extracts the identification times and time-to-trial's-end from the subject's actions during the trial. """
    if isinstance(identification_actions, SubjectActionTypesEnum):
        identification_actions = [identification_actions]
    identification_actions = list(set(identification_actions))
    actions = trial.get_actions()
    identification_times = (
        actions.loc[actions[cnfg.ACTION_STR].isin(identification_actions), cnfg.TIME_STR]
        .reset_index(drop=True)
        .rename(cnfg.TIME_STR)
    )
    # identification_times = actions.loc[np.isin(actions[cnfg.ACTION_STR], identification_actions)]
    # identification_times = identification_times.reset_index(drop=True)
    to_trial_end = (trial.end_time - identification_times).rename("to_trial_end")
    identification_times = pd.concat([identification_times, to_trial_end], axis=1)
    # return identification_times[cnfg.TIME_STR]
    return identification_times


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


def _classify_behavior(identifications: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    # find missed targets
    all_targets = targets.index
    missed_targets = all_targets[np.isin(all_targets, identifications[cnfg.TARGET_STR].unique(), invert=True)]
    misses = pd.DataFrame(index=range(len(missed_targets)))
    misses[cnfg.TARGET_STR] = missed_targets
    misses[[cnfg.TIME_STR, f"{cnfg.DISTANCE_STR}_px", f"{cnfg.DISTANCE_STR}_dva"]] = np.inf  # set unidentified times & dists to inf

    # append misses to the identifications DataFrame
    if not misses.empty:
        identifications = pd.concat([identifications, misses], axis=0)
    identifications = identifications.reset_index(drop=True, inplace=False)
    return identifications
