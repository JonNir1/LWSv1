import os
import warnings
from typing import Union, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm
import peyes

import config as cnfg
import helpers as hlp
from data_models.Subject import Subject
from data_models.Trial import Trial
from data_models.SearchArray import SearchArray
from data_models.LWSEnums import SubjectActionTypesEnum, TargetIdentificationTypeEnum
from pre_process.behavior import extract_trial_behavior

from pre_process.targets import extract_targets
from pre_process.metadata import extract_metadata
from pre_process.behavior import extract_behavior
from pre_process.fixations import extract_fixations
from pre_process.visits import extract_visits

_DEFAULT_IDENTIFICATION_ACTIONS = [
    SubjectActionTypesEnum.MARK_AND_CONFIRM,
    # SubjectActionTypesEnum.MARK_ONLY    # uncomment this to include marking-only actions
]


def process_subject(
        subj: Subject,
        identification_actions: Union[Sequence[SubjectActionTypesEnum], SubjectActionTypesEnum] = None,
        temporal_matching_threshold: float = cnfg.MAX_GAZE_TO_TRIGGER_TIME_DIFF,
        false_alarm_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
        visit_separation_time_threshold: float = cnfg.CHUNKING_TEMPORAL_WINDOW_MS,
        save_fixations: bool = True,
        verbose=False,
):
    identification_actions = identification_actions or _DEFAULT_IDENTIFICATION_ACTIONS
    bad_actions = [
        act for act in SubjectActionTypesEnum if act not in identification_actions and act != SubjectActionTypesEnum.NO_ACTION
    ]
    targets = extract_targets(subj)
    metadata = extract_metadata(subj, bad_actions=bad_actions)
    idents = extract_behavior(
        subj, identification_actions, temporal_matching_threshold, false_alarm_threshold_dva, verbose=verbose
    )
    fixations = extract_fixations(
        subj, identification_actions, temporal_matching_threshold, false_alarm_threshold_dva,
        save=save_fixations, verbose=verbose
    )
    visits = extract_visits(fixations, false_alarm_threshold_dva, visit_separation_time_threshold)
    return targets, metadata, idents, fixations, visits

