import os
from time import time
from typing import List, Union

import pandas as pd

import config as cnfg
from data_models.LWSEnums import SubjectActionCategoryEnum

from analysis.pipeline.preprocess_subjects import preprocess_all_subjects
from analysis.pipeline.extract_data import extract_data
from analysis.pipeline.funnels import append_funnels_to_fixations, append_funnels_to_visits

_DEFAULT_IDENTIFICATION_ACTIONS = [
    SubjectActionCategoryEnum.MARK_AND_CONFIRM,
    # SubjectActionCategoryEnum.MARK_ONLY    # uncomment this to include marking-only actions
]


def full_pipeline(
        raw_data_path: str = cnfg.RAW_DATA_PATH,
        identification_actions: Union[List[SubjectActionCategoryEnum], SubjectActionCategoryEnum] = None,
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
    identification_actions = identification_actions or _DEFAULT_IDENTIFICATION_ACTIONS
    subjects = preprocess_all_subjects(raw_data_path, verbose)
    targets, actions, metadata, idents, fixations, visits = extract_data(
        subjects,
        identification_actions=identification_actions,
        gaze_to_trigger_time_threshold=gaze_to_trigger_time_threshold,
        on_target_threshold_dva=on_target_threshold_dva,
        visit_merging_time_threshold=visit_merging_time_threshold,
        verbose=False,
    )
    fixations = append_funnels_to_fixations(
        fixations, metadata, idents,
        on_target_threshold_dva=on_target_threshold_dva,
        fixs_to_strip_threshold=cnfg.FIXATIONS_TO_STRIP_THRESHOLD,
        time_to_trial_end_threshold=cnfg.TIME_TO_TRIAL_END_THRESHOLD,
    )
    visits = append_funnels_to_visits(
        visits, metadata, idents, distance_type='min',  # can also be 'max' or 'weighted'
        on_target_threshold_dva=on_target_threshold_dva,
        fixs_to_strip_threshold=cnfg.FIXATIONS_TO_STRIP_THRESHOLD,
        time_to_trial_end_threshold=cnfg.TIME_TO_TRIAL_END_THRESHOLD,
    )
    if save:
        if verbose:
            print("Saving data to output path:", cnfg.OUTPUT_PATH)
        if not os.path.exists(cnfg.OUTPUT_PATH):
            os.makedirs(cnfg.OUTPUT_PATH)
        targets.to_pickle(os.path.join(cnfg.OUTPUT_PATH, 'targets.pkl'))
        actions.to_pickle(os.path.join(cnfg.OUTPUT_PATH, 'actions.pkl'))
        metadata.to_pickle(os.path.join(cnfg.OUTPUT_PATH, 'metadata.pkl'))
        idents.to_pickle(os.path.join(cnfg.OUTPUT_PATH, 'idents.pkl'))
        fixations.to_pickle(os.path.join(cnfg.OUTPUT_PATH, 'fixations.pkl'))
        visits.to_pickle(os.path.join(cnfg.OUTPUT_PATH, 'visits.pkl'))
    if verbose:
        print(f"Full pipeline completed in {time() - start_time:.2f} seconds.")
    return targets, actions, metadata, idents, fixations, visits


def read_saved_data(dir_path: str = cnfg.OUTPUT_PATH):
    try:
        targets = pd.read_pickle(os.path.join(dir_path, 'targets.pkl'))
    except FileNotFoundError:
        print("Targets data not found.")
        targets = None
    try:
        actions = pd.read_pickle(os.path.join(dir_path, 'actions.pkl'))
    except FileNotFoundError:
        print("Actions data not found.")
        actions = None
    try:
        metadata = pd.read_pickle(os.path.join(dir_path, 'metadata.pkl'))
    except FileNotFoundError:
        print("Metadata data not found.")
        metadata = None
    try:
        idents = pd.read_pickle(os.path.join(dir_path, 'idents.pkl'))
    except FileNotFoundError:
        print("Identifications data not found.")
        idents = None
    try:
        fixations = pd.read_pickle(os.path.join(dir_path, 'fixations.pkl'))
    except FileNotFoundError:
        print("Fixations data not found.")
        fixations = None
    try:
        visits = pd.read_pickle(os.path.join(dir_path, 'visits.pkl'))
    except FileNotFoundError:
        print("Visits data not found.")
        visits = None
    return targets, actions, metadata, idents, fixations, visits
