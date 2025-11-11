from typing import Union, List, Literal

import pandas as pd

import helpers.trial_exclusion as excl
from data_models.LWSEnums import SubjectActionCategoryEnum


def all_pass(event_data: pd.DataFrame) -> pd.Series:
    """ Dummy funnel-step that all events pass through. """
    return pd.Series(True, index=event_data.index, dtype=bool, name="all")


def trial_gaze_coverage(
        event_data: pd.DataFrame,
        metadata: pd.DataFrame,
        min_percent: Union[int, float],
) -> pd.Series:
    """ Funnel-step that passes events from trials with sufficient gaze coverage. """
    trial_has_coverage = excl.has_gaze_coverage(metadata, min_percent)  # boolean Series indexed by (subject, trial)
    has_gaze_coverage = event_data.apply(
        lambda row: trial_has_coverage.loc[(row["subject"], row["trial"])], axis=1,
    )
    has_gaze_coverage.rename("trial_gaze_coverage")
    return has_gaze_coverage


def trial_has_actions(
        event_data: pd.DataFrame,
        actions: pd.DataFrame,
        metadata: pd.DataFrame,
) -> pd.Series:
    """ Funnel-step that passes events from trials where the subject performed any actions. """
    trials_with_actions = excl.has_actions(actions, metadata)  # boolean Series indexed by (subject, trial)
    has_actions = event_data.apply(
        lambda row: trials_with_actions.loc[(row["subject"], row["trial"])], axis=1,
    )
    has_actions = has_actions.rename("trial_has_actions")
    return has_actions


def trial_no_bad_action(
        event_data: pd.DataFrame,
        actions: pd.DataFrame,
        metadata: pd.DataFrame,
        bad_actions: Union[SubjectActionCategoryEnum, List[SubjectActionCategoryEnum]],
) -> pd.Series:
    """ Funnel-step that passes events from trials where the subject performed no "bad" actions. """
    trial_no_bad_actions = excl.no_bad_actions(   # boolean Series indexed by (subject, trial)
        actions, metadata, bad_actions
    )
    has_no_bad_actions = event_data.apply(
        lambda row: trial_no_bad_actions.loc[(row["subject"], row["trial"])], axis=1,
    )
    has_no_bad_actions = has_no_bad_actions.rename("trial_no_bad_action")
    return has_no_bad_actions


def trial_no_false_alarm(event_data: pd.DataFrame, metadata: pd.DataFrame, idents: pd.DataFrame,) -> pd.Series:
    """ Funnel-step that passes events from trials where the subject made no false alarm identifications. """
    trial_no_false_alarms = excl.no_false_alarms(idents, metadata)  # boolean Series indexed by (subject, trial)
    no_false_alarms = event_data.apply(
        lambda row: trial_no_false_alarms.loc[(row["subject"], row["trial"])], axis=1,
    )
    no_false_alarms = no_false_alarms.rename("trial_no_false_alarm")
    return no_false_alarms


def instance_on_target(
        event_data: pd.DataFrame,
        on_target_threshold_dva: float,
        event_type: Literal["fixation", "visit"],
) -> pd.Series:
    """ Funnel-step that passes events where the instance is on-target. """
    if on_target_threshold_dva <= 0:
        raise ValueError(f"Parameter `on-target` threshold must be positive, got {on_target_threshold_dva}.")
    if event_type == "fixation":
        dist_columns = [
            # target0_distance_dva, target1_distance_dva, ...
            col for col in event_data.columns if col.startswith("target") and col.endswith("distance_dva")
        ]
    elif event_type == "visit":
        dist_columns = [
            # min_distance_dva, max_distance_dva, weighted_distance_dva
            # TODO: we use `any()` which is equivalent to specifying `min_distance_dva`; should this be explicit?
            col for col in event_data.columns if col.endswith("distance_dva")
        ]
    else:
        raise ValueError(f"Unknown event type: {event_type}. Expected 'fixation' or 'visit'.")
    is_on_target = event_data.apply(
        lambda row: any(row[dist_col] <= on_target_threshold_dva for dist_col in dist_columns),
        axis=1
    )
    is_on_target = is_on_target.rename(f"on_target")
    return is_on_target


def instance_not_outlier(
        event_data: pd.DataFrame, event_type: Literal["fixation", "visit"],
) -> pd.Series:
    """ Funnel-step that passes non-outlier fixations, or all visits. """
    if event_type == "fixation":
        not_outlier = event_data["outlier_reasons"].map(lambda val: len(val) == 0).rename("not_outlier")
    elif event_type == "visit":
        not_outlier = pd.Series(True, index=event_data.index, dtype=bool, name="not_outlier")
    else:
        raise ValueError(f"Unknown event type: {event_type}. Expected 'fixation' or 'visit'.")
    return not_outlier


def instance_before_identification(
        event_data: pd.DataFrame,
        idents: pd.DataFrame,
) -> pd.Series:
    """ Funnel-step that passes events ending before the subject's identification of the target. """
    before_identification = event_data.apply(
        lambda row: row["end_time"] < _find_identification_time(idents, row["subject"], row["trial"], row["target"]),
        axis=1
    )
    before_identification = before_identification.rename("before_identification")
    return before_identification


def instance_after_identification(
        event_data: pd.DataFrame,
        idents: pd.DataFrame,
) -> pd.Series:
    """ Funnel-step that passes events starting after the subject's identification of the target. """
    after_identification = event_data.apply(
        lambda row: row["start_time"] > _find_identification_time(idents, row["subject"], row["trial"], row["target"]),
        axis=1
    )
    after_identification = after_identification.rename("after_identification")
    return after_identification


def instance_not_close_to_trial_end(
        event_data: pd.DataFrame, time_to_trial_end_threshold: float,
) -> pd.Series:
    """ Funnel-step that passes events with sufficient time to trial end. """
    if time_to_trial_end_threshold < 0:
        raise ValueError(f"Minimum time to trial end must be non-negative, got {time_to_trial_end_threshold}.")
    not_close_to_end = event_data["to_trial_end"] >= time_to_trial_end_threshold
    not_close_to_end = not_close_to_end.rename("not_close_to_trial_end")
    return not_close_to_end


def instance_not_before_exemplar_visit(
        event_data: pd.DataFrame, exemplar_visit_threshold: int,
) -> pd.Series:
    """ Funnel-step that passes events that did not precede exemplar (bottom-strip) visit within the given number of
    fixations. """
    if exemplar_visit_threshold < 0:
        raise ValueError(f"Exemplar visit threshold must be non-negative, got {exemplar_visit_threshold}.")
    not_before_exemplar_visit = event_data["num_fixs_to_strip"] >= exemplar_visit_threshold
    not_before_exemplar_visit = not_before_exemplar_visit.rename("not_before_exemplar_visit")
    return not_before_exemplar_visit


def _find_identification_time(
        idents: pd.DataFrame, subj_id: int, trial_num: int, target: str
) -> float:
    is_subject = idents["subject"] == subj_id
    is_trial = idents["trial"] == trial_num
    is_target = idents["target"] == target
    return idents.loc[is_subject & is_trial & is_target, "time"].values[0]
