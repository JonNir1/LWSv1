from typing import List, Union

import numpy as np
import pandas as pd

from config import MILLISECONDS_IN_SECOND
from data_models.LWSEnums import SubjectActionCategoryEnum
from analysis.helpers.funnel import convert_criteria_to_funnel
from analysis.helpers.sdt import calc_sdt_class_per_trial


TRIAL_INCLUSION_STEPS = [
    # sequence of steps to determine if a trial is valid and included for further analysis
    "all",
    "gaze_coverage",
    "fixation_rate",
    # "has_actions",    # uncomment to exclude trials with no subject-actions
    "no_bad_action",
    "no_miss_with_false_alarm",
]


def check_trial_inclusion_criteria(
        metadata: pd.DataFrame,
        fixations: pd.DataFrame,
        actions: pd.DataFrame,
        idents: pd.DataFrame,
        min_gaze_coverage: int | float,
        min_fixation_rate: float,
        bad_actions: Union[SubjectActionCategoryEnum, List[SubjectActionCategoryEnum]],
        require_actions: bool,
        as_funnel: bool,
) -> pd.DataFrame:
    """
    Check trial inclusion criteria based on the specified steps in TRIAL_INCLUSION_STEPS.
    If `as_funnel` is True, converts the resulting criteria DataFrame into a funnel format where each step's column
    indicates whether the trial passed all criteria up to and including that step.

     - metadata: DataFrame containing trial metadata, including gaze coverage and duration.
     - fixations: DataFrame containing fixation data, used to calculate fixation rate.
     - actions: DataFrame containing subject actions, used to check for presence of actions and bad actions.
     - idents: DataFrame containing identification data, used to check for misses with false alarms.
     - min_gaze_coverage: Minimum percentage of trial time that gaze data must cover for a trial to be included.
     - min_fixation_rate: Minimum fixation rate (fixations per second) for a trial to be included.
     - bad_actions: SubjectActionCategoryEnum or list of enums indicating which actions are considered "bad" for exclusion.
     - require_actions: If True, trials with no subject-actions will be excluded; if False, they will be included.
     - as_funnel: If True, converts the resulting criteria DataFrame into a funnel format.

    Returns a DataFrame indexed by (subject, trial), with boolean columns for each inclusion step and a final
    "is_included" column indicating overall inclusion status.
    """
    ordered_components = []
    for step in TRIAL_INCLUSION_STEPS:
        if step == "all":
            all_trials = all_pass(metadata)
            ordered_components.append(all_trials)
        elif step == "gaze_coverage":
            has_coverage = has_gaze_coverage(metadata, min_gaze_coverage)
            ordered_components.append(has_coverage)
        elif step == "fixation_rate":
            has_high_rate = has_high_fixation_rate(fixations, metadata, min_fixation_rate)
            ordered_components.append(has_high_rate)
        elif step == "has_actions" and require_actions:
            has_acts = has_actions(actions, metadata)
            ordered_components.append(has_acts)
        elif step == "no_bad_action":
            no_bad_acts = no_bad_actions(actions, metadata, bad_actions)
            ordered_components.append(no_bad_acts)
        elif step == "no_miss_with_false_alarm":
            no_miss_w_fas = no_misses_with_false_alarms(idents, metadata)
            ordered_components.append(no_miss_w_fas)
        else:
            raise NotImplementedError(f"Unknown trial inclusion step: {step}")
    inclusion_df = (
        pd.concat(ordered_components, axis=1)
        .assign(is_included=lambda df: df.all(axis=1))
        .sort_index(level=["subject", "trial"])
        .astype(bool)
    )
    if as_funnel:
        inclusion_df = convert_criteria_to_funnel(inclusion_df)
    return inclusion_df


def all_pass(metadata: pd.DataFrame) -> pd.Series:
    """ Dummy inclusion criterion that all trials pass. """
    return pd.Series(True, index=metadata.set_index(["subject", "trial"]).index, name="all")


def has_gaze_coverage(metadata: pd.DataFrame, min_percent: int | float) -> pd.Series:
    """ Check if each trial has sufficient gaze coverage based on the provided minimum percentage threshold. """
    if min_percent <= 1.0:
        min_percent *= 100
    if not (0 < min_percent <= 100):
        raise ValueError("min_percent must be between 0 and 100.")
    gaze_coverage_percent = (
        metadata
        .set_index(["subject", "trial"])
        .loc[:, "gaze_coverage"]
    )
    has_coverage = (gaze_coverage_percent > min_percent).rename("has_gaze_coverage")
    return has_coverage


def has_high_fixation_rate(fixations: pd.DataFrame, metadata: pd.DataFrame, min_rate: float) -> pd.Series:
    """ Check if each trial has a high enough fixation rate based on the provided minimum rate threshold. """
    assert min_rate >= 0.0, "min_rate must be non-negative."
    fix_count = fixations.groupby(["subject", "trial", "eye"]).size().rename("n_fixations")
    fix_count = fix_count.groupby(["subject", "trial"]).max()   # take the max fixation count across both eyes for each trial
    trial_duration_sec = metadata.set_index(["subject", "trial"])["duration"] / MILLISECONDS_IN_SECOND
    has_high_rate = ((fix_count / trial_duration_sec) > min_rate).rename("has_high_fixation_rate")
    return has_high_rate


def has_actions(actions: pd.DataFrame, metadata: pd.DataFrame) -> pd.Series:
    """ Check if each trial has any actions (excluding NO_ACTION) recorded in the actions DataFrame. """
    trials_with_actions = list(
        actions
        .loc[actions["action"] != SubjectActionCategoryEnum.NO_ACTION, ["subject", "trial"]]
        .copy()
        .drop_duplicates()
        .itertuples(index=False)
    )
    trial_has_actions = (
        metadata
        .assign(has_actions=lambda df: df.apply(
            lambda row: (row["subject"], row["trial"]) in trials_with_actions, axis=1,
        ))
        .set_index(["subject", "trial"])
        .loc[:, "has_actions"]
    )
    trial_has_actions = trial_has_actions.rename("has_actions")
    return trial_has_actions


def no_bad_actions(
        actions: pd.DataFrame,
        metadata: pd.DataFrame,
        bad_actions: Union[SubjectActionCategoryEnum, List[SubjectActionCategoryEnum]],
) -> pd.Series:
    """
    Check if each trial has no bad actions (as defined in the bad_actions parameter) recorded in the actions DataFrame.
    """
    if isinstance(bad_actions, SubjectActionCategoryEnum):
        bad_actions = [bad_actions]
    bad_action_trials = list(
        actions.loc[np.isin(actions["action"], bad_actions), ["subject", "trial"]]
        .drop_duplicates()
        .itertuples(index=False)
    )
    trial_has_no_bad_actions = (
        metadata
        .assign(has_no_bad_actions=lambda df: df.apply(
            lambda row: (row["subject"], row["trial"]) not in bad_action_trials, axis=1,
        ))
        .set_index(["subject", "trial"])
        .loc[:, "has_no_bad_actions"]
    )
    trial_has_no_bad_actions = trial_has_no_bad_actions.rename("has_no_bad_actions")
    return trial_has_no_bad_actions


def no_misses_with_false_alarms(
        idents: pd.DataFrame,
        metadata: pd.DataFrame,
) -> pd.Series:
    """
    Check if each trial has no false alarms AND misses recorded in the idents DataFrame.
    If a trial has FAs and no misses - the subject was overly liberal with identification and we should keep the trial;
    but if the trial has FAs and a missed target - the subject may have confused the target with a distractor and
    we should exclude the trial.
    """
    misses = calc_sdt_class_per_trial(metadata, idents, "miss")
    false_alarms = calc_sdt_class_per_trial(metadata, idents, "false_alarm")
    has_misses_with_false_alarms = (misses["count"] > 0) & (false_alarms["count"] > 0)      # boolean Series
    trial_no_misses_with_false_alarms = pd.Series(
        ~has_misses_with_false_alarms.values,
        index=metadata.set_index(["subject", "trial"]).index,
        name="no_misses_with_false_alarms",
    )
    return trial_no_misses_with_false_alarms
