from typing import Literal

import numpy as np
import pandas as pd

import config as cnfg


def append_funnels_to_fixations(
        fixations: pd.DataFrame,
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        on_target_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
        fixs_to_strip_threshold: int = cnfg.FIXATIONS_TO_STRIP_THRESHOLD,
        time_to_trial_end_threshold: float = cnfg.TIME_TO_TRIAL_END_THRESHOLD,
) -> pd.DataFrame:
    return _append_funnels(
        fixations,
        metadata,
        idents,
        "fixation",
        on_target_threshold_dva,
        fixs_to_strip_threshold,
        time_to_trial_end_threshold
    )


def append_funnels_to_visits(
        visits: pd.DataFrame,
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        distance_type: Literal['min', 'max', 'weighted'],
        on_target_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
        fixs_to_strip_threshold: int = cnfg.FIXATIONS_TO_STRIP_THRESHOLD,
        time_to_trial_end_threshold: float = cnfg.TIME_TO_TRIAL_END_THRESHOLD,
) -> pd.DataFrame:
    distance_cols = [col for col in visits.columns if col == f"{distance_type}_{cnfg.DISTANCE_STR}_dva"]
    if len(distance_cols) == 0:
        raise KeyError(f"No distance columns found for `{distance_type}` distance type.")
    if len(distance_cols) > 1:
        raise KeyError(f"Multiple distance columns found for `{distance_type}` distance type: {distance_cols}.")
    distance_col = distance_cols[0]
    visits_copy = visits.copy()
    visits_copy[cnfg.DISTANCE_DVA_STR] = visits_copy[distance_col]
    visits = _append_funnels(
        visits_copy,
        metadata,
        idents,
        "visit",
        on_target_threshold_dva,
        fixs_to_strip_threshold,
        time_to_trial_end_threshold
    )
    visits = visits.drop(columns=[distance_col])    # drop the added distance column
    return visits



def _append_funnels(
        data: pd.DataFrame,
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        event_type: Literal["fixation", "visit"],
        on_target_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
        fixs_to_strip_threshold: int = cnfg.FIXATIONS_TO_STRIP_THRESHOLD,
        time_to_trial_end_threshold: float = cnfg.TIME_TO_TRIAL_END_THRESHOLD,
) -> pd.DataFrame:
    appended_columns = [cnfg.SUBJECT_STR, cnfg.TRIAL_STR, cnfg.EYE_STR, cnfg.TARGET_STR]
    if event_type == "fixation":
        appended_columns += [cnfg.EVENT_STR]
    elif event_type == "visit":
        appended_columns += [cnfg.VISIT_STR]
    else:
        raise ValueError(f"Unknown event type: {event_type}. Expected 'fixation' or 'visit'.")
    lws_funnel = _calculate_funnel(
        data, metadata, idents, event_type, "lws",
        on_target_threshold_dva, fixs_to_strip_threshold, time_to_trial_end_threshold
    )
    return_funnel = _calculate_funnel(
        data, metadata, idents, event_type, "target_return",
        on_target_threshold_dva, fixs_to_strip_threshold, time_to_trial_end_threshold
    )
    assert data.shape[0] == lws_funnel.shape[0] == return_funnel.shape[0], \
        f"{event_type.capitalize()} length {data.shape} does not match funnel lengths {lws_funnel.shape} and {return_funnel.shape}."
    return_funnel = return_funnel.drop(columns=[col for col in return_funnel.columns if col in lws_funnel.columns])  # dedup
    result = pd.concat([data, lws_funnel, return_funnel,], axis=1)
    result = (
        result
        .reset_index(drop=False)
        .sort_values(appended_columns)
        .reset_index(drop=True)
    )
    return result


def _calculate_funnel(
        data: pd.DataFrame,
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        event_type: Literal["fixation", "visit"],
        funnel_type: Literal["lws", "target_return"],
        on_target_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
        fixs_to_strip_threshold: int = cnfg.FIXATIONS_TO_STRIP_THRESHOLD,
        time_to_trial_end_threshold: float = cnfg.TIME_TO_TRIAL_END_THRESHOLD,
) -> pd.DataFrame:
    if funnel_type == "lws":
        funnel_steps = cnfg.LWS_FUNNEL_STEPS
    elif funnel_type == "target_return":
        funnel_steps = cnfg.TARGET_RETURN_FUNNEL_STEPS
    else:
        raise ValueError(f"Unknown funnel type: {funnel_type}. Expected 'lws' or 'target_return'.")
    if event_type == "visit":
        funnel_steps = [step for step in funnel_steps if step != "not_outlier"]

    results = dict()
    for step in funnel_steps:
        if step == cnfg.ALL_STR:
            step_res = pd.Series(True, index=data.index, dtype=bool)

        elif step == "valid_trial":
            step_res = data[cnfg.TRIAL_STR].map(
                lambda trial_num: not metadata.loc[metadata[cnfg.TRIAL_STR] == trial_num, f"bad_actions"].values[0]
            )

        elif step == "not_outlier":
            if event_type == "fixation":
                step_res = data["outlier_reasons"].map(lambda val: len(val) == 0)
            else:
                raise NotImplementedError(f"`not_outlier` step is not applicable for `visit` event type.")

        elif step == "on_target":
            assert on_target_threshold_dva > 0, f"On-target threshold must be positive, got {on_target_threshold_dva}."
            if event_type == "fixation":
                step_res = data.apply(
                    lambda row: row.loc[f"{row[cnfg.TARGET_STR]}_{cnfg.DISTANCE_STR}_dva"] <= on_target_threshold_dva,
                    axis=1
                )
            elif event_type == "visit":
                step_res = data[cnfg.DISTANCE_DVA_STR] <= on_target_threshold_dva

        elif step == "before_identification":
            step_res = data.apply(
                lambda row: row[cnfg.END_TIME_STR] < _find_identification_time(
                    idents, row[cnfg.SUBJECT_STR], row[cnfg.TRIAL_STR], row[cnfg.TARGET_STR]
                ),
                axis=1
            )

        elif step == "after_identification":
            step_res = data.apply(
                lambda row: row[cnfg.START_TIME_STR] > _find_identification_time(
                    idents, row[cnfg.SUBJECT_STR], row[cnfg.TRIAL_STR], row[cnfg.TARGET_STR]
                ),
                axis=1
            )

        elif step == "fixs_to_strip":
            assert fixs_to_strip_threshold >= 0, f"Fixations to strip threshold must be non-negative, got {fixs_to_strip_threshold}."
            step_res = data["num_fixs_to_strip"] > fixs_to_strip_threshold

        elif step == "not_end_with_trial":
            assert time_to_trial_end_threshold >= 0, f"Minimum time to trial end must be non-negative, got {time_to_trial_end_threshold}."
            step_res = data["to_trial_end"] >= time_to_trial_end_threshold

        elif step == "is_lws" or step == "is_return":
            step_res = pd.Series(np.array([results[s] for s in funnel_steps if not s.startswith("is_")]).all(axis=0))

        else:
            raise ValueError(f"Unknown funnel step: {step}.")
        results[step] = step_res.rename(step)
    funnel_df = pd.concat(results.values(), keys=results.keys(), axis=1).astype(bool)
    return funnel_df


def _find_identification_time(
        idents: pd.DataFrame, subj_id: int, trial_num: int, target: str
) -> float:
    is_subject = idents[cnfg.SUBJECT_STR] == subj_id
    is_trial = idents[cnfg.TRIAL_STR] == trial_num
    is_target = idents[cnfg.TARGET_STR] == target
    return idents.loc[is_subject & is_trial & is_target, cnfg.TIME_STR].values[0]
