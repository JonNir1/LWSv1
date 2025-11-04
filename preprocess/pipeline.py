import os
from time import time
from typing import List, Union

import pandas as pd

import config as cnfg
from data_models.LWSEnums import SubjectActionCategoryEnum

from preprocess.parse_raw_data import parse_all_subjects
from preprocess.build_dataframes import build_dataframes


def full_pipeline(
        raw_data_path: str = cnfg.RAW_DATA_PATH,
        identification_actions: Union[SubjectActionCategoryEnum, List[SubjectActionCategoryEnum]] = cnfg.IDENTIFICATION_ACTIONS,
        gaze_to_trigger_time_threshold: float = cnfg.MAX_GAZE_TO_TRIGGER_TIME_DIFF,
        on_target_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
        visit_merging_time_threshold: float = cnfg.VISIT_MERGING_TIME_THRESHOLD,
        save: bool = True,
        verbose: bool = True,
) -> (
        pd.DataFrame,   # targets
        pd.DataFrame,   # actions
        pd.DataFrame,   # metadata
        pd.DataFrame,   # identifications
        pd.DataFrame,   # fixations
        pd.DataFrame,   # visits
):
    start_time = time()
    if isinstance(identification_actions, SubjectActionCategoryEnum):
        identification_actions = [identification_actions]
    if not identification_actions:
        raise ValueError(f"Must specify actions for argument `identification_actions`.")
    if gaze_to_trigger_time_threshold < 0:
        raise ValueError(f"`gaze_to_trigger_time_threshold` must be non-negative.")
    if on_target_threshold_dva < 0:
        raise ValueError(f"`on_target_threshold_dva` must be non-negative.")
    if visit_merging_time_threshold < 0:
        raise ValueError(f"`visit_merging_time_threshold` must be non-negative.")
    subjects = parse_all_subjects(raw_data_path, verbose)
    targets, actions, metadata, idents, fixations, visits = build_dataframes(
        subjects,
        identification_actions=identification_actions,
        gaze_to_trigger_time_threshold=gaze_to_trigger_time_threshold,
        on_target_threshold_dva=on_target_threshold_dva,
        visit_merging_time_threshold=visit_merging_time_threshold,
        verbose=False,
    )
    if save:
        save_to = cnfg.OUTPUT_PATH
        if verbose:
            print("Saving data to output path:", save_to)
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        targets.to_pickle(os.path.join(save_to, 'targets.pkl'))
        actions.to_pickle(os.path.join(save_to, 'actions.pkl'))
        metadata.to_pickle(os.path.join(save_to, 'metadata.pkl'))
        idents.to_pickle(os.path.join(save_to, 'idents.pkl'))
        fixations.to_pickle(os.path.join(save_to, 'fixations.pkl'))
        visits.to_pickle(os.path.join(save_to, 'visits.pkl'))
    if verbose:
        print(f"Full pipeline completed in {time() - start_time:.2f} seconds.")
    return targets, actions, metadata, idents, fixations, visits



