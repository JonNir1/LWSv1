from typing import List, Union, Callable

import pandas as pd

from config import MILLISECONDS_IN_SECOND
from analysis.helpers.funnels.funnel_config import TRIAL_INCLUSION_CRITERIA
from data_models.LWSEnums import SubjectActionCategoryEnum
from analysis.helpers.sdt import calc_sdt_class_per_trial

_SUBJECT_TRIAL_COLS = ["subject", "trial"]


def check_trial_inclusion_criteria(
        metadata: pd.DataFrame,
        fixations: pd.DataFrame,
        actions: pd.DataFrame,
        idents: pd.DataFrame,
        min_gaze_coverage: int | float,
        min_fixation_rate: float,
        bad_actions: Union[SubjectActionCategoryEnum, List[SubjectActionCategoryEnum]],
        require_actions: bool,
) -> pd.DataFrame:
    """ Returns a DataFrame indexed by (subject, trial) with boolean columns for each criterion and `is_valid_trial`. """
    meta_idx = metadata.set_index(_SUBJECT_TRIAL_COLS).index
    # criterion registry: each callable returns a boolean Series indexed by (subject, trial)
    criteria_functions: dict[str, Callable[[], pd.Series]] = {
        "gaze_coverage": lambda: has_gaze_coverage(metadata, min_gaze_coverage),
        "fixation_rate": lambda: has_high_fixation_rate(fixations, metadata, min_fixation_rate),
        "has_actions": lambda: has_actions(actions, meta_idx),
        "no_bad_action": lambda: no_bad_actions(actions, meta_idx, bad_actions),
        "no_miss_with_false_alarm": lambda: no_misses_with_false_alarms(idents, metadata),
    }
    ordered_components: list[pd.Series] = []
    for crit in TRIAL_INCLUSION_CRITERIA:
        if crit == "has_actions" and not require_actions:
            continue
        if crit not in criteria_functions.keys():
            raise NotImplementedError(f"Unknown trial inclusion step: {crit}")
        ordered_components.append(criteria_functions[crit]())
    inclusion_df = (
        pd.concat(ordered_components, axis=1)
        .reindex(meta_idx)  # ensure same order / includes duplicates if any
        .assign(is_valid_trial=lambda df: df.all(axis=1))
        .sort_index(level=_SUBJECT_TRIAL_COLS)
        .astype(bool)
    )
    return inclusion_df


def all_pass(trial_indices: pd.MultiIndex) -> pd.Series:
    """ Dummy inclusion criterion that all trials pass. """
    return pd.Series(True, index=trial_indices, name="all")


def has_gaze_coverage(metadata: pd.DataFrame, min_percent: int | float) -> pd.Series:
    """ True iff trial has at least `min_percent` gaze coverage (percent of trial time with gaze data). """
    if min_percent <= 1.0:
        min_percent *= 100
    if not (0 < min_percent <= 100):
        raise ValueError("min_percent must be between 0 and 100.")
    gaze_coverage_percent = metadata.set_index(_SUBJECT_TRIAL_COLS)["gaze_coverage"]
    return (gaze_coverage_percent > min_percent).rename("gaze_coverage")


def has_high_fixation_rate(fixations: pd.DataFrame, metadata: pd.DataFrame, min_rate: float) -> pd.Series:
    """True iff trial has a high enough fixation rate (fixations per second)."""
    assert min_rate >= 0.0, "min_rate must be non-negative."
    fix_count = (
        fixations
        .groupby(["subject", "trial", "eye"]).size()
        .groupby(_SUBJECT_TRIAL_COLS)
        .max()  # take the max fixation count across both eyes for each trial
        .rename("n_fixations")
    )
    trial_duration_sec = metadata.set_index(["subject", "trial"])["duration"] / MILLISECONDS_IN_SECOND
    return ((fix_count / trial_duration_sec) > min_rate).rename("fixation_rate")


def has_actions(actions: pd.DataFrame, trial_indices: pd.MultiIndex) -> pd.Series:
    """True iff trial has any action other than NO_ACTION."""
    trials_with_actions_idxs = _trials_index_from_actions(
        actions, actions["action"] != SubjectActionCategoryEnum.NO_ACTION
    )
    return pd.Series(trial_indices.isin(trials_with_actions_idxs), index=trial_indices, name="has_actions")


def no_bad_actions(
    actions: pd.DataFrame,
    trial_indices: pd.MultiIndex,
    bad_actions: Union[SubjectActionCategoryEnum, List[SubjectActionCategoryEnum]],
) -> pd.Series:
    """True iff trial has no actions designated as `bad_actions`."""
    if isinstance(bad_actions, SubjectActionCategoryEnum):
        bad_actions = [bad_actions]
    bad_action_trial_idxs = _trials_index_from_actions(actions, actions["action"].isin(bad_actions))
    return pd.Series(
        ~trial_indices.isin(bad_action_trial_idxs), index=trial_indices, name="no_bad_action"
    )


def no_misses_with_false_alarms(
        idents: pd.DataFrame,
        metadata: pd.DataFrame,
) -> pd.Series:
    """ True iff NOT (misses>0 AND false_alarms>0) for that trial. """
    misses = calc_sdt_class_per_trial(metadata, idents, "miss")
    false_alarms = calc_sdt_class_per_trial(metadata, idents, "false_alarm")
    has_misses_with_false_alarms = (misses["count"] > 0) & (false_alarms["count"] > 0)      # boolean Series
    trial_no_misses_with_false_alarms = pd.Series(
        ~has_misses_with_false_alarms.values,
        index=metadata.set_index(["subject", "trial"]).index,
        name="no_miss_with_false_alarm",
    )
    return trial_no_misses_with_false_alarms


def _trials_index_from_actions(actions: pd.DataFrame, mask: pd.Series) -> pd.MultiIndex:
    """Helper: build a MultiIndex of (subject, trial) for rows where mask is True."""
    return pd.MultiIndex.from_frame(actions.loc[mask, _SUBJECT_TRIAL_COLS].drop_duplicates())
