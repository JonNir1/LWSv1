import os
from time import time
from typing import List, Union, Literal

import pandas as pd
from tqdm import tqdm

import config as cnfg
from data_models.Subject import Subject
from data_models.LWSEnums import SubjectActionTypesEnum

from analysis.pipeline.preprocess_subjects import preprocess_all_subjects
from analysis.pipeline.extract_data import extract_data
from analysis.pipeline.lws_funnel import fixation_funnel, visit_funnel

_DEFAULT_IDENTIFICATION_ACTIONS = [
    SubjectActionTypesEnum.MARK_AND_CONFIRM,
    # SubjectActionTypesEnum.MARK_ONLY    # uncomment this to include marking-only actions
]


def full_pipeline(
        raw_data_path: str = cnfg.RAW_DATA_PATH,
        identification_actions: Union[List[SubjectActionTypesEnum], SubjectActionTypesEnum] = None,
        gaze_to_trigger_time_threshold: float = cnfg.MAX_GAZE_TO_TRIGGER_TIME_DIFF,
        on_target_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
        visit_merging_time_threshold: float = cnfg.VISIT_MERGING_TIME_THRESHOLD,
        fixs_to_strip_threshold: int = cnfg.FIXATIONS_TO_STRIP_THRESHOLD,
        time_to_trial_end_threshold: float = cnfg.TIME_TO_TRIAL_END_THRESHOLD,
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

    lws_fixation_funnel = fixation_funnel(
        fixations, metadata, idents, on_target_threshold_dva, fixs_to_strip_threshold, time_to_trial_end_threshold
    )
    fixations = pd.concat([fixations, lws_fixation_funnel], axis=1)
    lws_visit_funnel = visit_funnel(
        visits, metadata, idents, on_target_threshold_dva, fixs_to_strip_threshold, time_to_trial_end_threshold,
        distance_type='min',    # `min`, `max`, or `weighted` can be used here
    )
    visits = pd.concat([visits, lws_visit_funnel], axis=1)
    if verbose:
        print(f"Full pipeline completed in {time() - start_time:.2f} seconds.")
    return targets, actions, metadata, idents, fixations, visits
