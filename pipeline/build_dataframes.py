from time import time
from typing import List, Union, Literal

import pandas as pd
from tqdm import tqdm

import config as cnfg

from data_models.Subject import Subject
from data_models.LWSEnums import SubjectActionCategoryEnum


def build_dataframes(
        subjects: List[Subject],
        identification_actions: Union[List[SubjectActionCategoryEnum], SubjectActionCategoryEnum],
        gaze_to_trigger_time_threshold: float,
        on_target_threshold_dva: float,
        verbose=False,
) -> (
        pd.DataFrame,   # icons
        pd.DataFrame,   # actions
        pd.DataFrame,   # metadata
        pd.DataFrame,   # identifications
        pd.DataFrame,   # eye movements
):
    start_time = time()
    bad_actions = [
        act for act in SubjectActionCategoryEnum if
        act not in identification_actions and act != SubjectActionCategoryEnum.NO_ACTION
    ]
    icons = _concat_subject_results(subjects, "icon", verbose=verbose)
    # `pd.concat` widens categoricals with differing categories back to object; re-apply so icons.pkl stays small
    for col in (cnfg.ICON_STR, "sub_path", cnfg.CATEGORY_STR):
        if col in icons.columns:
            icons[col] = icons[col].astype("category")
    actions = _concat_subject_results(subjects, "action", verbose=verbose)
    metadata = _concat_subject_results(subjects, "metadata", bad_actions=bad_actions, verbose=verbose,)
    idents = _concat_subject_results(
        subjects,
        "identification",
        identification_actions=identification_actions,
        gaze_to_trigger_match_threshold=gaze_to_trigger_time_threshold,
        on_target_threshold_dva=on_target_threshold_dva,
        verbose=verbose,
    )
    eye_movements = _concat_subject_results(subjects, "event", verbose=verbose,)
    # `pd.concat` widens categoricals with differing categories back to object; re-apply so the pickle stays small
    eye_movements[f"{cnfg.EVENT_STR}_type"] = eye_movements[f"{cnfg.EVENT_STR}_type"].astype("category")
    if verbose:
        print(f"Data extraction completed in {time() - start_time:.2f} seconds.")
    return icons, actions, metadata, idents, eye_movements


def _concat_subject_results(
        subjects: List[Subject],
        to_concat: Literal["icon", "action", "metadata", "identification", "event"],
        verbose: bool = True,
        **kwargs
) -> pd.DataFrame:
    results = dict()
    for subj in tqdm(subjects, desc=f"Extracting {to_concat} data", disable=not verbose):
        if to_concat == "icon":
            subj_res = subj.get_icons()
        elif to_concat == "action":
            subj_res = subj.get_actions()
        elif to_concat == "metadata":
            bad_actions = kwargs.get("bad_actions", [])
            assert bad_actions, f"Must specify `bad_actions` for `{to_concat}` concatenation."
            subj_res = subj.get_metadata(bad_actions)
        elif to_concat == "identification":
            identification_actions = kwargs.get("identification_actions", None)
            assert identification_actions, f"Must specify `identification_actions` for `{to_concat}` concatenation."
            gaze_to_trigger_match_threshold = kwargs.get("gaze_to_trigger_match_threshold", None)
            assert gaze_to_trigger_match_threshold and gaze_to_trigger_match_threshold > 0, \
                f"Must specify positive `gaze_to_trigger_match_threshold` for `{to_concat}` concatenation."
            on_target_threshold_dva = kwargs.get("on_target_threshold_dva", None)
            assert on_target_threshold_dva and on_target_threshold_dva > 0, \
                f"Must specify positive `on_target_threshold_dva` for `{to_concat}` concatenation."
            subj_res = subj.get_target_identifications(
                identification_actions, gaze_to_trigger_match_threshold, on_target_threshold_dva, verbose=False,
            )
        elif to_concat == "event":
            subj_res = subj.get_events(save=True, verbose=verbose)
        else:
            raise ValueError(f"Unknown type: {to_concat}")
        results[subj.id] = subj_res
    results = (
        pd.concat(results.values(), keys=results.keys(), axis=0)
        .reset_index(drop=False)
        .rename(columns={"level_0": "subject"})
        .drop(columns=["level_1"])
        .sort_values(by=["subject", "trial"])
        .reset_index(drop=True)
    )
    return results
