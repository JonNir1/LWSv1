from typing import Dict, Optional

import pandas as pd

import config as cnfg


FUNNEL_STEPS = [
    "all",
    "valid_trials",
    "not_outlier",  # non-outlier fixations
    "on_target",
    "before_identification",
    "next_1_not_in_strip",
    "next_2_not_in_strip",
    "next_3_not_in_strip",
    "not_end_with_trial"
]


def calc_funnel(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calculate the LWS funnel steps from the provided DataFrame.
    The DataFrame should contain all necessary columns for the funnel calculation.
    """
    funnel = dict()
    funnel['all'] = data
    for step in FUNNEL_STEPS:
        funnel[step] = _calc_funnel_step(funnel, step)
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
