from typing import Literal, List

import numpy as np
import pandas as pd

import config as cnfg

LWS_FUNNEL_STEPS = [
    # sequence of steps to determine if a fixation/visit is a Looking-without-Seeing (LWS) instance
    "all", "valid_trial", "not_outlier", "on_target", "before_identification", "fixs_to_strip", "not_end_with_trial", "is_lws"
]
TARGET_RETURN_FUNNEL_STEPS = [
    # sequence of steps to determine if a fixation/visit is a post-identification target-return instance
    "all", "valid_trial", "not_outlier", "on_target", "after_identification", "is_return"
]


def calc_funnel_sizes(data: pd.DataFrame) -> pd.DataFrame:
    """ Calculates the size of each LWS funnel-step for each subject and trial in the provided data. """
    assert cnfg.SUBJECT_STR in data.columns, f"Data must contain `{cnfg.SUBJECT_STR}` column."
    assert cnfg.TRIAL_STR in data.columns, f"Data must contain `{cnfg.TRIAL_STR}` column."
    steps = [step for step in LWS_FUNNEL_STEPS if step in data.columns]
    sizes = dict()
    for (subj, trial), group in data.groupby([cnfg.SUBJECT_STR, cnfg.TRIAL_STR]):
        for i, curr_step in enumerate(steps):
            curr_and_prev_steps = steps[:i + 1]
            step_size = group[curr_and_prev_steps].all(axis=1).sum()
            sizes[(subj, trial, curr_step)] = step_size
    sizes = (
        pd.Series(sizes)
        .reset_index(drop=False)
        .rename(columns={"level_0": cnfg.SUBJECT_STR, "level_1": cnfg.TRIAL_STR, "level_2": "step", 0: "size"})
        .sort_values(by=["step"], key=lambda steps_series: steps_series.map(lambda step: LWS_FUNNEL_STEPS.index(step)))
        .sort_values(by=[cnfg.SUBJECT_STR, cnfg.TRIAL_STR])
        .reset_index(drop=True)
    )
    return sizes


def fixation_funnel(
        fixations: pd.DataFrame,
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        on_target_threshold_dva: float,
        fixs_to_strip_threshold: int,
        time_to_trial_end_threshold: float
) -> pd.DataFrame:
    return _lws_funnel(
        fixations,
        "fixations",
        metadata,
        idents,
        on_target_threshold_dva,
        fixs_to_strip_threshold,
        time_to_trial_end_threshold
    )


def visit_funnel(
        visits: pd.DataFrame,
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        on_target_threshold_dva: float,
        fixs_to_strip_threshold: int,
        time_to_trial_end_threshold: float,
        distance_type: Literal['min', 'max', 'weighted']
) -> pd.DataFrame:
    distance_cols = [col for col in visits.columns if col == f"{distance_type}_{cnfg.DISTANCE_STR}_dva"]
    if len(distance_cols) == 0:
        raise KeyError(f"No distance columns found for `{distance_type}` distance type.")
    if len(distance_cols) > 1:
        raise KeyError(f"Multiple distance columns found for `{distance_type}` distance type: {distance_cols}.")
    visits_copy = visits.copy()
    visits_copy[cnfg.DISTANCE_DVA_STR] = visits_copy[distance_cols[0]]
    return _lws_funnel(
        visits_copy,
        "visits",
        metadata,
        idents,
        on_target_threshold_dva,
        fixs_to_strip_threshold,
        time_to_trial_end_threshold
    )


def lws_funnel(
        data: pd.DataFrame,
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        event_type: Literal["fixations", "visits"],
        on_target_threshold_dva: float,
        fixs_to_strip_threshold: int,
        time_to_trial_end_threshold: float
) -> pd.DataFrame:
    assert on_target_threshold_dva > 0, f"On-target threshold must be positive, got {on_target_threshold_dva}."
    assert fixs_to_strip_threshold >= 0, f"Fixations to strip threshold must be non-negative, got {fixs_to_strip_threshold}."
    assert time_to_trial_end_threshold >= 0, f"Minimum time to trial end must be non-negative, got {time_to_trial_end_threshold}."
    appended_columns = [cnfg.SUBJECT_STR, cnfg.TRIAL_STR, cnfg.EYE_STR]
    if event_type == "fixations":
        funnel_steps = LWS_FUNNEL_STEPS
        appended_columns += [cnfg.EVENT_STR]
    elif event_type == "visits":
        funnel_steps = [step for step in LWS_FUNNEL_STEPS if step != "not_outlier"]
        appended_columns += [cnfg.VISIT_STR]
    else:
        raise ValueError(f"Unknown event type: {event_type}. Expected 'fixations' or 'visits'.")
    funnel_df = _calculate_funnel(
        data, metadata, idents, funnel_steps, event_type,
        on_target_threshold_dva=on_target_threshold_dva,
        fixs_to_strip_threshold=fixs_to_strip_threshold,
        time_to_trial_end_threshold=time_to_trial_end_threshold
    )
    funnel_df = pd.concat([data[appended_columns], funnel_df], axis=1)
    return funnel_df


def _calculate_funnel(
        data: pd.DataFrame,
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        funnel_steps: List[str],
        event_type: Literal["fixations", "visits"],
        on_target_threshold_dva: float = None,
        fixs_to_strip_threshold: int = None,
        time_to_trial_end_threshold: float = None,
) -> pd.DataFrame:
    results = dict()
    for step in funnel_steps:
        if step == cnfg.ALL_STR:
            step_res = pd.Series(True, index=data.index, dtype=bool)
        elif step == "valid_trial":
            step_res = data[cnfg.TRIAL_STR].map(
                lambda trial_num: not metadata.loc[metadata[cnfg.TRIAL_STR] == trial_num, f"bad_actions"].values[0]
            )
        elif step == "not_outlier":
            if event_type == "fixations":
                step_res = data["outlier_reasons"].map(lambda val: len(val) == 0)
            else:
                raise NotImplementedError(f"`not_outlier` step is not applicable for `visits` funnel type.")
        elif step == "on_target":
            assert on_target_threshold_dva > 0, f"On-target threshold must be positive, got {on_target_threshold_dva}."
            if event_type == "fixations":
                step_res = data.apply(
                    lambda row: row.loc[f"{row[cnfg.TARGET_STR]}_{cnfg.DISTANCE_STR}_dva"] <= on_target_threshold_dva,
                    axis=1
                )
            elif event_type == "visits":
                step_res = data[cnfg.DISTANCE_DVA_STR] <= on_target_threshold_dva
        elif step == "before_identification":
            step_res = data.apply(
                lambda row: row[cnfg.END_TIME_STR] < _find_identification_time(idents, row[cnfg.TRIAL_STR], row[cnfg.TARGET_STR]),
                axis=1
            )
        elif step == "after_identification":
            step_res = data.apply(
                lambda row: row[cnfg.START_TIME_STR] > _find_identification_time(idents, row[cnfg.TRIAL_STR], row[cnfg.TARGET_STR]),
                axis=1
            )
        elif step == "fixs_to_strip":
            assert fixs_to_strip_threshold >= 0, f"Fixations to strip threshold must be non-negative, got {fixs_to_strip_threshold}."
            step_res = data["num_fixs_to_strip"] > fixs_to_strip_threshold
        elif step == "not_end_with_trial":
            assert time_to_trial_end_threshold >= 0, f"Minimum time to trial end must be non-negative, got {time_to_trial_end_threshold}."
            step_res = data["to_trial_end"] >= time_to_trial_end_threshold
        elif step == "is_lws" or step == "is_return":
            step_res = pd.Series(np.array([results[s] for s in funnel_steps if s != "is_lws"]).all(axis=0))
        else:
            raise ValueError(f"Unknown funnel step: {step}.")
        step_res.name = step
        results[step] = step_res
    funnel_df = pd.concat(results.values(), keys=results.keys(), axis=1).astype(bool)
    return funnel_df






def _find_identification_time(idents: pd.DataFrame, trial_num: int, target: str) -> float:
    is_trial = idents[cnfg.TRIAL_STR] == trial_num
    is_target = idents[cnfg.TARGET_STR] == target
    return idents.loc[is_trial & is_target, cnfg.TIME_STR].values[0]



