from typing import List, Union

import numpy as np
import pandas as pd

import config as cnfg
from data_models.LWSEnums import SubjectActionCategoryEnum
from helpers.sdt import calc_sdt_class_per_trial


def check_exclusion_criteria(
        metadata: pd.DataFrame,
        actions: pd.DataFrame,
        idents: pd.DataFrame,
        min_percent: int | float = cnfg.GAZE_COVERAGE_PERCENT_THRESHOLD,
        bad_actions: Union[SubjectActionCategoryEnum, List[SubjectActionCategoryEnum]] = cnfg.BAD_ACTIONS,
        require_actions: bool = False,
) -> pd.DataFrame:
    has_coverage = has_gaze_coverage(metadata, min_percent)
    no_bad_acts = no_bad_actions(actions, metadata, bad_actions)
    no_fas = no_false_alarms(idents, metadata)
    components = [has_coverage, no_bad_acts, no_fas]
    if require_actions:
        has_acts = has_actions(actions, metadata)
        components.append(has_acts)
    criteria_df = (
        pd.concat(components, axis=1)
        .assign(is_included=lambda df: df.all(axis=1))
        .reset_index(drop=False)
        .sort_values(["subject", "trial"])
        .reset_index(drop=True)
    )
    return criteria_df


def has_gaze_coverage(metadata: pd.DataFrame, min_percent: int | float) -> pd.Series:
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


def has_actions(actions: pd.DataFrame, metadata: pd.DataFrame) -> pd.Series:
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


def no_false_alarms(
        idents: pd.DataFrame,
        metadata: pd.DataFrame,
) -> pd.Series:
    false_alarms = calc_sdt_class_per_trial(metadata, idents, "false_alarm")
    trial_no_false_alarms = (
        false_alarms
        .set_index(["subject", "trial"])
        .assign(no_false_alarms=lambda df: df["count"] == 0)
        .loc[:, "no_false_alarms"]
    )
    trial_no_false_alarms = trial_no_false_alarms.rename("no_false_alarms")
    return trial_no_false_alarms
