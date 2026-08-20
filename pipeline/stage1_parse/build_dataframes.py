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
        verbose=False,
) -> (
        pd.DataFrame,   # icons
        pd.DataFrame,   # actions
        pd.DataFrame,   # metadata
        pd.DataFrame,   # eye movements
):
    start_time = time()
    bad_actions = [
        act for act in SubjectActionCategoryEnum if
        act not in identification_actions and act != SubjectActionCategoryEnum.NO_ACTION
    ]
    icons = _concat_subject_results(subjects, "icon", verbose=verbose)
    for col in (cnfg.ICON_STR, "sub_path", cnfg.CATEGORY_STR):
        if col in icons.columns:
            icons[col] = icons[col].astype("category")
    actions = _concat_subject_results(subjects, "action", verbose=verbose)
    metadata = _concat_subject_results(subjects, "metadata", bad_actions=bad_actions, verbose=verbose,)
    eye_movements = _concat_subject_results(subjects, "event", verbose=verbose,)
    eye_movements[f"{cnfg.EVENT_STR}_type"] = eye_movements[f"{cnfg.EVENT_STR}_type"].astype("category")
    if verbose:
        print(f"Data extraction completed in {time() - start_time:.2f} seconds.")
    return icons, actions, metadata, eye_movements


def _concat_subject_results(
        subjects: List[Subject],
        to_concat: Literal["icon", "action", "metadata", "event"],
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
