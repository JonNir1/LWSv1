from typing import Literal

import pandas as pd

from analysis.helpers.funnel import convert_criteria_to_funnel

IS_LWS_STEPS = [
    "not_outlier", "on_target", "before_identification", "not_close_to_trial_end", "not_before_exemplar_visit",
]
IS_TARGET_RETURN_STEPS = [
    "not_outlier", "on_target", "after_identification",
]


def check_lws_criteria(
        event_data: pd.DataFrame,
        idents: pd.DataFrame,
        on_target_threshold_dva: float,
        time_to_trial_end_threshold: float,
        min_fixs_from_exemplars: int,
        event_type: Literal["fixation", "visit"],
        as_funnel: bool,
) -> pd.DataFrame:
    """
    Check Looking-without-Seeing (LWS) criteria for each fixation/visit in the event data, based on the specified steps
    in IS_LWS_STEPS. if `as_funnel` is True, converts the resulting criteria DataFrame into a funnel format where each
    step's column indicates whether the event passed all criteria up to and including that step.

     - event_data: DataFrame containing fixation or visit data, including relevant columns for each criterion step.
     - idents: DataFrame containing identification data, used to check for identification times.
     - on_target_threshold_dva: Distance threshold in degrees of visual angle for determining if an event is on-target.
     - time_to_trial_end_threshold: Minimum time in seconds that an event must end before trial end to be considered LWS.
     - min_fixs_from_exemplars: Minimum number of fixations that must occur between an event and subsequent
     exemplar-section fixation for it to be considered LWS.
     - event_type: String indicating whether the events are "fixation" or "visit", which determines how certain
     criteria are evaluated.
     - as_funnel: If True, converts the resulting criteria DataFrame into a funnel format.

    Returns a DataFrame with the same index as `event_data`, boolean columns for each LWS criterion step, and a final
    "is_lws" column indicating overall LWS status for each event.
    """
    ordered_components = []
    for step in IS_LWS_STEPS:
        if step == "not_outlier":
            not_outlier = is_not_outlier(event_data, event_type)
            ordered_components.append(not_outlier)
        elif step == "on_target":
            on_target = is_on_target(event_data, on_target_threshold_dva, event_type)
            ordered_components.append(on_target)
        elif step == "before_identification":
            before_identification = is_before_identification(event_data, idents)
            ordered_components.append(before_identification)
        elif step == "not_close_to_trial_end":
            not_close_to_end = is_not_close_to_trial_end(event_data, time_to_trial_end_threshold)
            ordered_components.append(not_close_to_end)
        elif step == "not_before_exemplar_visit":
            not_before_exemplar_visit = is_not_before_exemplar_fixation(event_data, min_fixs_from_exemplars)
            ordered_components.append(not_before_exemplar_visit)
        else:
            raise NotImplementedError(f"Unknown LWS criterion step: {step}")
    lws_criterion_df = (
        pd.concat(ordered_components, axis=1)
        .assign(is_lws=lambda df: df.all(axis=1))
        .astype(bool)
    )
    lws_criterion_df.index = event_data.index  # ensure the index matches the original event data
    if as_funnel:
        lws_criterion_df = convert_criteria_to_funnel(lws_criterion_df)
    return lws_criterion_df


def check_target_return_criteria(
        event_data: pd.DataFrame,
        idents: pd.DataFrame,
        on_target_threshold_dva: float,
        event_type: Literal["fixation", "visit"],
        as_funnel: bool,
) -> pd.DataFrame:
    """
    Check target-return criteria for each fixation/visit in the event data, based on the specified steps
    in IS_TARGET_RETURN_STEPS. if `as_funnel` is True, converts the resulting criteria DataFrame into a funnel format where each
    step's column indicates whether the event passed all criteria up to and including that step.

     - event_data: DataFrame containing fixation or visit data, including relevant columns for each criterion step.
     - idents: DataFrame containing identification data, used to check for identification times.
     - on_target_threshold_dva: Distance threshold in degrees of visual angle for determining if an event is on-target.
     - event_type: String indicating whether the events are "fixation" or "visit", which determines how certain
     criteria are evaluated.
     - as_funnel: If True, converts the resulting criteria DataFrame into a funnel format.

    Returns a DataFrame with the same index as `event_data`, boolean columns for each Target-Return criterion step, and
    a final "is_target_return" column indicating overall target-return status for each event.
    """
    ordered_components = []
    for step in IS_TARGET_RETURN_STEPS:
        if step == "not_outlier":
            not_outlier = is_not_outlier(event_data, event_type)
            ordered_components.append(not_outlier)
        elif step == "on_target":
            on_target = is_on_target(event_data, on_target_threshold_dva, event_type)
            ordered_components.append(on_target)
        elif step == "after_identification":
            after_identification = is_after_identification(event_data, idents)
            ordered_components.append(after_identification)
        else:
            raise NotImplementedError(f"Unknown target-return criterion step: {step}")
    target_return_criterion_df = pd.concat(ordered_components, axis=1).assign(
        is_target_return=lambda df: df.all(axis=1)
    )
    target_return_criterion_df.index = event_data.index     # ensure the index matches the original event data
    if as_funnel:
        target_return_criterion_df = convert_criteria_to_funnel(target_return_criterion_df)
    return target_return_criterion_df


def is_not_outlier(event_data: pd.DataFrame, event_type: Literal["fixation", "visit"]) -> pd.Series:
    """ Check if fixation is not an outlier based on the `outlier_reasons` column, or return True for all visits. """
    if event_type == "fixation":
        not_outlier = event_data["outlier_reasons"].map(lambda val: len(val) == 0).rename("not_outlier")
    elif event_type == "visit":
        not_outlier = pd.Series(True, index=event_data.index, dtype=bool, name="not_outlier")
    else:
        raise ValueError(f"Unknown event type: {event_type}. Expected 'fixation' or 'visit'.")
    return not_outlier


def is_on_target(
        event_data: pd.DataFrame, on_target_threshold_dva: float, event_type: Literal["fixation", "visit"],
) -> pd.Series:
    """ Check if fixation/visit has a minimum distance from target below the specified threshold """
    if on_target_threshold_dva <= 0:
        raise ValueError(f"Parameter `on-target` threshold must be positive, got {on_target_threshold_dva}.")
    if event_type == "fixation":
        dist_columns = [
            col for col in event_data.columns if col.startswith("target") and col.endswith("distance_dva")
            # target0_distance_dva, target1_distance_dva, ...
        ]
    elif event_type == "visit":
        dist_columns = [
            # TODO: we use `any()` which is equivalent to specifying `min_distance_dva`; should this be explicit?
            col for col in event_data.columns if col.endswith("distance_dva")
            # min_distance_dva, max_distance_dva, weighted_distance_dva
        ]
    else:
        raise ValueError(f"Unknown event type: {event_type}. Expected 'fixation' or 'visit'.")
    on_target = event_data.apply(
        lambda row: any(row[dist_col] <= on_target_threshold_dva for dist_col in dist_columns),
        axis=1
    )
    on_target = on_target.rename(f"on_target")
    return on_target


def is_before_identification(event_data: pd.DataFrame, idents: pd.DataFrame) -> pd.Series:
    """ Check if fixation/visit ends before the subject's identification of the target. """
    before_identification = event_data.apply(
        lambda row: row["end_time"] < _find_identification_time(idents, row["subject"], row["trial"], row["target"]),
        axis=1
    )
    before_identification = before_identification.rename("before_identification")
    return before_identification


def is_after_identification(event_data: pd.DataFrame, idents: pd.DataFrame) -> pd.Series:
    """ Check if fixation/visit starts after the subject's identification of the target. """
    after_identification = event_data.apply(
        lambda row: row["start_time"] > _find_identification_time(idents, row["subject"], row["trial"], row["target"]),
        axis=1
    )
    after_identification = after_identification.rename("after_identification")
    return after_identification


def is_not_close_to_trial_end(event_data: pd.DataFrame, time_to_trial_end_threshold: float) -> pd.Series:
    """ Check if fixation/visit ends with sufficient time remaining before trial end """
    if time_to_trial_end_threshold < 0:
        raise ValueError(f"Minimum time to trial end must be non-negative, got {time_to_trial_end_threshold}.")
    not_close_to_end = event_data["to_trial_end"] >= time_to_trial_end_threshold
    not_close_to_end = not_close_to_end.rename("not_close_to_trial_end")
    return not_close_to_end


def is_not_before_exemplar_fixation(event_data: pd.DataFrame, min_fixs_from_exemplars: int) -> pd.Series:
    """ Check if fixation/visit didn't occur at least `min_fixs_from_strip` before a fixation in the exemplar section """
    if min_fixs_from_exemplars < 0:
        raise ValueError(
            f"Minimum number of fixations from exemplars must be non-negative, got {min_fixs_from_exemplars}."
        )
    not_before_exemplar_visit = event_data["num_fixs_to_strip"] >= min_fixs_from_exemplars
    not_before_exemplar_visit = not_before_exemplar_visit.rename("not_before_exemplar_visit")
    return not_before_exemplar_visit


def _find_identification_time(idents: pd.DataFrame, subj_id: int, trial_num: int, target: str) -> float:
    is_subject = idents["subject"] == subj_id
    is_trial = idents["trial"] == trial_num
    is_target = idents["target"] == target
    return idents.loc[is_subject & is_trial & is_target, "time"].values[0]
