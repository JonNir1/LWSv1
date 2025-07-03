from typing import Literal, List, Dict

import numpy as np
import pandas as pd

import config as cnfg

LWS_FUNNEL_STEPS = [
    "all", "valid_trial", "not_outlier", "on_target", "before_identification", "fixs_to_strip", "not_end_with_trial"
]


def fixation_funnel(
        fixations: pd.DataFrame,
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        on_target_threshold_dva: float,
        fixs_to_strip_threshold: int,
        time_from_trial_end_threshold: float
) -> pd.DataFrame:
    return _lws_funnel(
        fixations,
        "fixations",
        metadata,
        idents,
        on_target_threshold_dva,
        fixs_to_strip_threshold,
        time_from_trial_end_threshold
    )


def visit_funnel(
        visits: pd.DataFrame,
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        on_target_threshold_dva: float,
        fixs_to_strip_threshold: int,
        time_from_trial_end_threshold: float,
        distance_type: Literal['min', 'max', 'weighted']
) -> pd.DataFrame:
    distance_cols = [col for col in visits.columns if col == f"{distance_type}_{cnfg.DISTANCE_STR}_dva"]
    if len(distance_cols) == 0:
        raise KeyError(f"No distance columns found for `{distance_type}` distance type.")
    if len(distance_cols) > 1:
        raise KeyError(f"Multiple distance columns found for `{distance_type}` distance type: {distance_cols}.")
    visits_copy = visits.copy()
    visits_copy[f"{cnfg.DISTANCE_STR}_dva"] = visits_copy[distance_cols[0]]
    return _lws_funnel(
        visits_copy,
        "visits",
        metadata,
        idents,
        on_target_threshold_dva,
        fixs_to_strip_threshold,
        time_from_trial_end_threshold
    )


def _lws_funnel(
        data: pd.DataFrame,
        funnel_type: Literal["fixations", "visits"],
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        on_target_threshold_dva: float,
        fixs_to_strip_threshold: int,
        time_from_trial_end_threshold: float
) -> pd.DataFrame:
    assert on_target_threshold_dva > 0, f"On-target threshold must be positive, got {on_target_threshold_dva}."
    assert time_from_trial_end_threshold >= 0, f"Minimum time from trial end must be non-negative, got {time_from_trial_end_threshold}."
    if funnel_type == "fixations":
        funnel_steps = LWS_FUNNEL_STEPS
    elif funnel_type == "visits":
        funnel_steps = [step for step in LWS_FUNNEL_STEPS if step != "not_outlier"]
    else:
        raise ValueError(f"Unknown funnel type: {funnel_type}. Expected 'fixations' or 'visits'.")

    results = dict()
    for step in funnel_steps:
        if step == cnfg.ALL_STR:
            step_res = pd.Series(True, index=data.index, dtype=bool)
        elif step == "valid_trial":
            step_res = data[cnfg.TRIAL_STR].map(
                lambda trial_num: not metadata.loc[metadata[cnfg.TRIAL_STR] == trial_num, f"bad_actions"].values[0]
            )
        elif step == "not_outlier":
            if funnel_type == "fixations":
                step_res = data["outlier_reasons"].map(lambda val: len(val) == 0)
            else:
                raise NotImplementedError(f"`not_outlier` step is not applicable for `visits` funnel type.")
        elif step == "on_target":
            if funnel_type == "fixations":
                step_res = data.apply(lambda row: row.loc[f"{row[cnfg.TARGET_STR]}_{cnfg.DISTANCE_STR}_dva"], axis=1)
            elif funnel_type == "visits":
                step_res = data[f"{cnfg.DISTANCE_STR}_dva"] <= on_target_threshold_dva
        elif step == "before_identification":
            step_res = data.apply(
                lambda row: row[cnfg.END_TIME_STR] <= _find_identification_time(idents, row[cnfg.TRIAL_STR], row[cnfg.TARGET_STR]),
                axis=1
            )
        elif step == "fixs_to_strip":
            step_res = data["num_fixs_to_strip"] > fixs_to_strip_threshold
        elif step == "not_end_with_trial":
            step_res = data["to_trial_end"] >= time_from_trial_end_threshold
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



