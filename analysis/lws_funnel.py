from typing import Dict, List, Literal

import pandas as pd

import config as cnfg


FUNNEL_STEPS = [
    "all",
    "valid_trials",
    "not_outlier",  # non-outlier fixations
    "on_target",
    "before_identification",
    "fixs_to_strip",
    "not_end_with_trial"
]


def fixation_funnel(
        fixations: pd.DataFrame,
        min_fixs_to_strip: int = 3,
        min_time_to_trial_end: float = cnfg.CHUNKING_TEMPORAL_WINDOW_MS,
) -> Dict[str, pd.DataFrame]:
    # TODO: exclude on-target fixations that are within K fixations from the target identification
    return _calc_funnel(
        data=fixations,
        min_fixs_to_strip=min_fixs_to_strip,
        min_time_to_trial_end = min_time_to_trial_end,
        steps=FUNNEL_STEPS,
        distance_col=cnfg.DISTANCE_STR,
    )


def visit_funnel(
        visits: pd.DataFrame,
        distance_col: Literal['min_distance', 'max_distance', 'mean_distance', 'weighted_distance'],
        min_fixs_to_strip: int = 3,
        min_time_to_trial_end: float = cnfg.CHUNKING_TEMPORAL_WINDOW_MS,
) -> Dict[str, pd.DataFrame]:
    return _calc_funnel(
        data=visits,
        distance_col=distance_col,
        min_fixs_to_strip=min_fixs_to_strip,
        min_time_to_trial_end=min_time_to_trial_end,
        steps=[step for step in FUNNEL_STEPS if step != "not_outlier"],
    )


def _calc_funnel(
        data: pd.DataFrame, steps: List[str],
        distance_col: str,
        min_fixs_to_strip: int,
        min_time_to_trial_end: float,
) -> Dict[str, pd.DataFrame]:
    assert steps[0] == cnfg.ALL_STR, f"First step must be `{cnfg.ALL_STR}`."
    assert distance_col in data.columns, f"Distance column `{distance_col}` not found in data."
    assert min_fixs_to_strip >= 0, "Minimum number of fixations to strip must be non-negative."
    assert min_time_to_trial_end >= 0, "Minimum time to trial end must be non-negative."
    funnel = dict()
    funnel[cnfg.ALL_STR] = data
    for i, step in enumerate(steps):
        if i == 0:
            continue
        prev_result = funnel[steps[i - 1]]
        if step == "valid_trials":
            funnel[step] = prev_result[~prev_result[f"bad_{cnfg.TRIAL_STR}"].astype(bool)]
            continue
        if step == "not_outlier":
            funnel[step] = prev_result[prev_result["outlier_reasons"].map(lambda val: len(val) == 0)]
            continue
        if step == "on_target":
            funnel[step] = prev_result[prev_result[distance_col] <= cnfg.ON_TARGET_THRESHOLD_DVA]
            continue
        if step == "before_identification":
            funnel[step] = prev_result[prev_result[f"{cnfg.END_TIME_STR}"] <= prev_result[cnfg.TARGET_TIME_STR]]
            continue
        if step == "fixs_to_strip":
            funnel[f"{step}>={min_fixs_to_strip}"] = prev_result[prev_result["num_fixs_to_strip"] >= min_fixs_to_strip]
            continue
        if step == "not_end_with_trial":
            funnel[step] = prev_result[prev_result["to_trial_end"] >= min_time_to_trial_end]
            continue
        raise KeyError(f"Unknown funnel step: {step}.")
    return funnel


def _calc_funnel_step(funnel: Dict[str, pd.DataFrame], step: str) -> pd.DataFrame:
    assert cnfg.ALL_STR in funnel.keys(), "Funnel must contain the `all` step."
    if step not in FUNNEL_STEPS:
        raise ValueError(f"Unknown funnel step: {step}.")
    if step in funnel.keys():
        return funnel[step]
    prev_step = FUNNEL_STEPS[FUNNEL_STEPS.index(step) - 1]
    try:
        prev_result = funnel[prev_step]
    except KeyError:
        prev_result = _calc_funnel_step(funnel, prev_step)
    if step == "valid_trials":
        return prev_result[~prev_result[f"bad_{cnfg.TRIAL_STR}"].astype(bool)]
    if step == "not_outlier":
        return prev_result[prev_result["outlier_reasons"].map(lambda val: len(val) == 0)]
    if step == "on_target":
        return prev_result[prev_result[cnfg.DISTANCE_STR] <= cnfg.ON_TARGET_THRESHOLD_DVA]
    if step == "before_identification":
        return prev_result[prev_result[f"{cnfg.END_TIME_STR}"] <= prev_result[cnfg.TARGET_TIME_STR]]
    if step == "next_1_not_in_strip":
        return prev_result[~(prev_result['next_1_in_strip'].astype(bool))]
    if step == "next_2_not_in_strip":
        return prev_result[
            ~(prev_result['next_1_in_strip'].astype(bool)) &
            ~(prev_result['next_2_in_strip'].astype(bool))
        ]
    if step == "next_3_not_in_strip":
        return prev_result[
            ~(prev_result['next_1_in_strip'].astype(bool)) &
            ~(prev_result['next_2_in_strip'].astype(bool)) &
            ~(prev_result['next_3_in_strip'].astype(bool))
        ]
    if step == "not_end_with_trial":
        return prev_result[prev_result["to_trial_end"] >= cnfg.CHUNKING_TEMPORAL_WINDOW_MS]
    raise ValueError(f"Unknown funnel step: {step}.")
