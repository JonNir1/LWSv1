from typing import List, Literal, Union

import pandas as pd
from tqdm import tqdm

from data_models.LWSEnums import SubjectActionCategoryEnum
import funnel.steps as stp


def run_funnel(
        event_data: pd.DataFrame,
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        actions: pd.DataFrame,
        steps: List[str],
        event_type: Literal["fixation", "visit"],
        bad_actions: Union[SubjectActionCategoryEnum, List[SubjectActionCategoryEnum]],
        on_target_threshold_dva: float,
        time_to_trial_end_threshold: float,
        exemplar_visit_threshold: int,
        verbose: bool = False,
) -> pd.DataFrame:
    """ Runs the provided funnel-step functions on the event data and appends the results as new columns. """
    if event_type not in ["fixation", "visit"]:
        raise ValueError(f"Unknown event type: {event_type}. Expected 'fixation' or 'visit'.")
    appended_columns = ["subject", "trial", "eye", "target"] + ["event" if event_type == "fixation" else "visit"]
    results = dict()
    for step in tqdm(steps, desc=f"{event_type.capitalize()} Funnel Steps", disable=not verbose):
        if step == "all":
            results[step] = stp.all_pass(event_data)
        elif step == "trial_gaze_coverage":
            results[step] = stp.trial_gaze_coverage(event_data, metadata)
        elif step == "trial_no_bad_action":
            results[step] = stp.trial_no_bad_action(event_data, actions, bad_actions)
        elif step == "trial_no_false_alarm":
            results[step] = stp.trial_no_false_alarm(event_data, metadata, idents)
        elif "on_target" in step:
            results[step] = stp.instance_on_target(event_data, on_target_threshold_dva, event_type)
        elif "not_outlier" in step:
            results[step] = stp.instance_not_outlier(event_data, event_type)
        elif "before_identification" in step:
            results[step] = stp.instance_before_identification(event_data, idents)
        elif "after_identification" in step:
            results[step] = stp.instance_after_identification(event_data, idents)
        elif "to_trial_end" in step:
            results[step] = stp.instance_not_close_to_trial_end(event_data, time_to_trial_end_threshold)
        elif "bottom_strip" in step or "exemplar_visit" in step:
            results[step] = stp.instance_not_before_exemplar_visit(event_data, exemplar_visit_threshold)
        elif step in ["is_lws", "is_target_return"]:
            # these are calculated after all steps are run
            continue
        else:
            raise ValueError(f"Unknown funnel step: {step}")
    funnel_df = pd.concat(results.values(), keys=results.keys(), axis=1).astype(bool)
    final_df = (
        pd.concat([event_data[appended_columns], funnel_df], axis=1)
        .sort_values(appended_columns)
        .reset_index(drop=True)
        .set_index(appended_columns)
    )
    return final_df


def calculate_funnel_sizes(funnel_data: pd.DataFrame) -> pd.DataFrame:
    """ Calculates the size of each funnel-step for each subject and trial in the provided funnel_data. """
    assert "subject" in funnel_data.index.names, f"Funnel Data must contain index level `subject`."
    assert "trial" in funnel_data.index.names, f"Funnel Data must contain index level `trial`."
    assert "eye" in funnel_data.index.names, f"Funnel Data must contain index level `eye`."
    steps = list(funnel_data.columns)
    sizes = dict()
    for (subj, trial, eye), group in funnel_data.groupby(["subject", "trial", "eye"]):
        for i, curr_step in enumerate(steps):
            curr_and_prev_steps = steps[:i + 1]
            step_size = group[curr_and_prev_steps].all(axis=1).sum()
            sizes[(subj, trial, eye, curr_step)] = step_size
    sizes = (
        pd.Series(sizes)
        .reset_index(drop=False)
        .rename(columns={
            "level_0": "subject", "level_1": "trial", "level_2": "eye", "level_3": "step", 0: "size"
        })
        .sort_values(by=["step"], key=lambda steps_series: steps_series.map(lambda step: steps.index(step)))
        .sort_values(by=["subject", "trial", "eye"])
        .reset_index(drop=True)
    )
    return sizes
