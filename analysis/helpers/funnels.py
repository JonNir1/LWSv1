from typing import Literal

import pandas as pd

import config as cnfg


def calc_funnel_sizes(data: pd.DataFrame, funnel_type: Literal["lws", "target_return"],) -> pd.DataFrame:
    """ Calculates the size of each funnel-step for each subject and trial in the provided data. """
    assert cnfg.SUBJECT_STR in data.columns, f"Data must contain `{cnfg.SUBJECT_STR}` column."
    assert cnfg.TRIAL_STR in data.columns, f"Data must contain `{cnfg.TRIAL_STR}` column."
    funnel_steps = cnfg.LWS_FUNNEL_STEPS if funnel_type == "lws" else cnfg.TARGET_RETURN_FUNNEL_STEPS
    funnel_steps = [step for step in funnel_steps if step in data.columns]

    sizes = dict()
    for (subj, trial), group in data.groupby([cnfg.SUBJECT_STR, cnfg.TRIAL_STR]):
        for i, curr_step in enumerate(funnel_steps):
            curr_and_prev_steps = funnel_steps[:i + 1]
            step_size = group[curr_and_prev_steps].all(axis=1).sum()
            sizes[(subj, trial, curr_step)] = step_size
    sizes = (
        pd.Series(sizes)
        .reset_index(drop=False)
        .rename(columns={"level_0": cnfg.SUBJECT_STR, "level_1": cnfg.TRIAL_STR, "level_2": "step", 0: "size"})
        .sort_values(by=["step"], key=lambda steps_series: steps_series.map(lambda step: funnel_steps.index(step)))
        .sort_values(by=[cnfg.SUBJECT_STR, cnfg.TRIAL_STR])
        .reset_index(drop=True)
    )
    return sizes

