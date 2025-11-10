from typing import Literal, List, Union

import pandas as pd
from tqdm import tqdm

import config as cnfg
from pipeline.read_data import read_saved_data
from data_models.LWSEnums import SubjectActionCategoryEnum

import funnel.steps as stp


def prepare_funnel(
        data_dir: str,
        event_type: Literal["fixation", "visit"],
        funnel_type: Literal["lws", "target_return"],
        verbose: bool = True,
        **funnel_kwargs,
) -> pd.DataFrame:
    """
    Prepares the funnel results by:
    1) Reading saved data from the specified directory.
    2) Appending metadata to the event data (fixations or visits).
    3) Running the specified funnel analysis (LWS or Target-Return) on the event data.

    :param data_dir: Directory containing the saved data files.
    :param event_type: Type of event to analyze ("fixation" or "visit").
    :param funnel_type: Type of funnel analysis to perform ("lws" or "target_return").
    :param verbose: Whether to display progress bars during processing.

    :keyword bad_actions: list of subject actions considered as 'bad', resulting in excluding the trial from analysis.
    :keyword on_target_threshold_dva: threshold (in degrees of visual angle) to determine if an event is on-target.
    :keyword time_to_trial_end_threshold: (only for LWS funnel) threshold (in milliseconds) to determine if an event
    is too close to trial end.
    :keyword exemplar_visit_threshold: (only for LWS funnel) number of subsequent visits to the exemplar section
    (bottom strip) to consider when determining LWS status.
    """
    if event_type not in ["fixation", "visit"]:
        raise ValueError(f"Invalid `event_type`: {event_type}. Must be 'fixation' or 'visit'.")
    if funnel_type not in ["lws", "target_return"]:
        raise NotImplementedError(f"Unknown `funnel_type`: {funnel_type}.")
    targets, actions, metadata, idents, fixations, visits = read_saved_data(data_dir)
    event_data = fixations if event_type == "fixation" else visits
    event_data = _append_metadata_and_filter_by_eye(event_data, targets, metadata)
    if funnel_type == "lws":
        funnel_steps = cnfg.LWS_FUNNEL_STEPS
        time_to_trial_end_threshold = funnel_kwargs.get("time_to_trial_end_threshold", cnfg.TIME_TO_TRIAL_END_THRESHOLD)
        exemplar_visit_threshold = funnel_kwargs.get("exemplar_visit_threshold", cnfg.FIXATIONS_TO_STRIP_THRESHOLD)
    else:   # funnel_type == "target_return"
        funnel_steps = cnfg.TARGET_RETURN_FUNNEL_STEPS
        time_to_trial_end_threshold = 0     # not used in target-return funnel
        exemplar_visit_threshold = 0        # not used in target-return funnel
    funnel_step_results = _run_funnel_steps(
        event_data=event_data,
        metadata=metadata,
        idents=idents,
        actions=actions,
        event_type=event_type,
        funnel_steps=funnel_steps,
        bad_actions=funnel_kwargs.get("bad_actions", cnfg.BAD_ACTIONS),
        on_target_threshold_dva=funnel_kwargs.get("on_target_threshold_dva", cnfg.ON_TARGET_THRESHOLD_DVA),
        time_to_trial_end_threshold=time_to_trial_end_threshold,
        exemplar_visit_threshold=exemplar_visit_threshold,
        verbose=verbose
    )
    funnel_results = _apply_funnel(funnel_step_results, funnel_steps, verbose)
    return funnel_results


def _append_metadata_and_filter_by_eye(
        event_data: pd.DataFrame, targets: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """ Append trial & target metadata to event data and filter out irrelevant-eye data. """
    event_data = (
        event_data
        .copy()     # avoid modifying original data
        .merge(     # append trial category & dominant eye
            metadata[["subject", "trial", "trial_category", "dominant_eye"]],
            on=["subject", "trial"],
            how="left"
        )
        .merge(     # append target category & angle
            targets[["subject", "trial", "target", "category", "angle"]],
            on=["subject", "trial", "target"],
            how="left"
        )
        .rename(columns={"category": "target_category", "angle": "target_angle"})
        .astype({
            "subject": "category",
            "trial": "int64",
            "trial_category": "category",
            "target_category": "category",
            "target_angle": "float64",
        })
    )
    event_data = (  # filter out irrelevant-eye data
        event_data
        .loc[event_data["eye"].map(str.lower) == event_data["dominant_eye"].map(str.lower)]
        .drop(columns=["eye", "dominant_eye"])
    )
    return event_data


def _run_funnel_steps(
        event_data: pd.DataFrame,
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        actions: pd.DataFrame,
        event_type: Literal["fixation", "visit"],
        funnel_steps: List[str],
        bad_actions: Union[SubjectActionCategoryEnum, List[SubjectActionCategoryEnum]],
        on_target_threshold_dva: float,
        time_to_trial_end_threshold: float,
        exemplar_visit_threshold: int,
        verbose: bool = False,
) -> pd.DataFrame:
    """ Runs the provided funnel-step functions on the data and returns a DataFrame with the a column per step. """
    if event_type not in ["fixation", "visit"]:
        raise ValueError(f"Unknown event type: {event_type}. Expected 'fixation' or 'visit'.")
    appended_columns = ["subject", "trial", "target", "trial_category", "target_category", "target_angle"] + [
        "event" if event_type == "fixation" else "visit"
    ]
    missing_columns = [col for col in appended_columns if col not in event_data.columns]
    if missing_columns:
        raise ValueError(f"Event data is missing required columns: {missing_columns}")
    results = dict()
    for step in tqdm(funnel_steps, desc=f"Calculating Funnel Steps", disable=not verbose):
        if step == "all":
            results[step] = stp.all_pass(event_data)
        elif step == "trial_gaze_coverage":
            results[step] = stp.trial_gaze_coverage(event_data, metadata)
        elif step == "trial_has_actions":
            results[step] = stp.trial_has_actions(event_data, actions)
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
        elif step == "final":
            # calculated after all steps are run
            continue
        else:
            raise ValueError(f"Unknown funnel step: {step}")
    funnel_df = (
        pd.concat(results.values(), keys=results.keys(), axis=1)
        .assign(final=lambda df: df.all(axis=1))
        .astype(bool)
    )
    final_df = (
        pd.concat([event_data[appended_columns], funnel_df], axis=1)
        .sort_values(appended_columns)
        .reset_index(drop=True)
    )
    return final_df


def _apply_funnel(step_results: pd.DataFrame, funnel_steps: List[str], verbose: bool = False) -> pd.DataFrame:
    """ For each step in the funnel, check if it and all previous steps are True. """
    if "subject" not in step_results.columns:
        raise ValueError(f"`step_results` DataFrame must contain `subject` column.")
    if "trial" not in step_results.columns:
        raise ValueError(f"`step_results` DataFrame must contain `trial` column.")
    if "target" not in step_results.columns:
        raise ValueError(f"`step_results` DataFrame must contain `target` column.")
    missing_steps = [step for step in funnel_steps if step not in step_results.columns]
    if missing_steps:
        raise ValueError(f"`step_results` is missing required funnel step columns: {missing_steps}.")
    results = dict()
    for (subj, trl, tgt), group in tqdm(
            step_results.groupby(["subject", "trial", "target"]), desc="Applying Funnel", disable=not verbose
    ):
        group_copy = group.copy()
        for i, curr_step in enumerate(funnel_steps):
            curr_and_prev_steps = funnel_steps[:i + 1]
            group_copy[curr_step] = group_copy[curr_and_prev_steps].all(axis=1)
        results[(subj, trl, tgt)] = group_copy
    results = (
        pd.concat(results, ignore_index=True)
        .loc[   # reorder columns: first non-funnel columns, then funnel columns ordered by their step order
            :, [col for col in step_results.columns if col not in funnel_steps] +
               [col for col in step_results.columns if col in funnel_steps]
        ]
        .sort_values(by=["subject", "trial", "target"])
        .reset_index(drop=True)
    )
    return results

