from time import time
from typing import List, Union, Literal

import pandas as pd
from tqdm import tqdm

import config as cnfg
from data_models.Subject import Subject
from data_models.LWSEnums import SubjectActionTypesEnum

_DEFAULT_IDENTIFICATION_ACTIONS = [
    SubjectActionTypesEnum.MARK_AND_CONFIRM,
    # SubjectActionTypesEnum.MARK_ONLY    # uncomment this to include marking-only actions
]

def extract_data(
        subjects: List[Subject],
        identification_actions: Union[List[SubjectActionTypesEnum], SubjectActionTypesEnum],
        gaze_to_trigger_time_threshold: float,
        on_target_threshold_dva: float,
        visit_merging_time_threshold: float,
        verbose=False,
) -> (
        pd.DataFrame,   # targets
        pd.DataFrame,   # metadata
        pd.DataFrame,   # identifications
        pd.DataFrame,   # fixations
        pd.DataFrame,   # visits
):
    start_time = time()
    identification_actions = identification_actions or _DEFAULT_IDENTIFICATION_ACTIONS
    bad_actions = [
        act for act in SubjectActionTypesEnum if
        act not in identification_actions and act != SubjectActionTypesEnum.NO_ACTION
    ]
    targets = _concat_subject_results(subjects, cnfg.TARGET_STR, verbose=verbose)
    metadata = _concat_subject_results(subjects, cnfg.METADATA_STR, bad_actions=bad_actions, verbose=verbose,)
    idents = _concat_subject_results(
        subjects,
        "identification",
        identification_actions=identification_actions,
        gaze_to_trigger_match_threshold=gaze_to_trigger_time_threshold,
        verbose=verbose,
    )
    fixations = _concat_subject_results(subjects, cnfg.FIXATION_STR, verbose=verbose,)
    visits = _concat_subject_results(
        subjects,
        cnfg.VISIT_STR,
        on_target_threshold_dva=on_target_threshold_dva,
        visit_merging_time_threshold=visit_merging_time_threshold,
        verbose=verbose,
    )
    if verbose:
        print(f"Data extraction completed in {time() - start_time:.2f} seconds.")
    return targets, metadata, idents, fixations, visits


def _concat_subject_results(
        subjects: List[Subject],
        to_concat: Literal["target", "metadata", "identification", "fixation", "visit"],
        verbose: bool = True,
        **kwargs
) -> pd.DataFrame:
    results = dict()
    for subj in tqdm(subjects, desc=f"Extracting {to_concat} data", disable=not verbose):
        if to_concat == cnfg.TARGET_STR:
            subj_res = subj.get_targets()
        elif to_concat == cnfg.METADATA_STR:
            bad_actions = kwargs.get("bad_actions", None)
            assert bad_actions, f"Must specify `bad_actions` for `{to_concat}` concatenation."
            subj_res = subj.get_metadata(bad_actions)
        elif to_concat == "identification":
            identification_actions = kwargs.get("identification_actions", None)
            assert identification_actions, f"Must specify `identification_actions` for `{to_concat}` concatenation."
            gaze_to_trigger_match_threshold = kwargs.get("gaze_to_trigger_match_threshold", None)
            assert gaze_to_trigger_match_threshold and gaze_to_trigger_match_threshold > 0, \
                f"Must specify positive `gaze_to_trigger_match_threshold` for `{to_concat}` concatenation."
            subj_res = subj.get_target_identifications(
                identification_actions, gaze_to_trigger_match_threshold, verbose=False,
            )
        elif to_concat == cnfg.FIXATION_STR:
            subj_res = subj.get_fixations(save=True, verbose=verbose)
        elif to_concat == cnfg.VISIT_STR:
            on_target_threshold_dva = kwargs.get("on_target_threshold_dva", None)
            assert on_target_threshold_dva and on_target_threshold_dva > 0, \
                f"Must specify positive `on_target_threshold_dva` for `{to_concat}` concatenation."
            visit_merging_time_threshold = kwargs.get("visit_merging_time_threshold", None)
            assert visit_merging_time_threshold and visit_merging_time_threshold > 0, \
                f"Must specify positive `visit_merging_time_threshold` for `{to_concat}` concatenation."
            subj_res = subj.get_visits(on_target_threshold_dva, visit_merging_time_threshold,)
        else:
            raise ValueError(f"Unknown type: {to_concat}")
        results[subj.id] = subj_res
    results = (
        pd.concat(results.values(), keys=results.keys(), axis=0)
        .reset_index(drop=False)
        .rename(columns={"level_0": cnfg.SUBJECT_STR})
        .drop(columns=["level_1"])
    )
    return results
