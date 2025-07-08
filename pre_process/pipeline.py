import os
import warnings
from typing import Optional, Union, Sequence

import pandas as pd

import config as cnfg
from data_models.Subject import Subject
from data_models.LWSEnums import SubjectActionTypesEnum

from pre_process.targets import extract_targets
from pre_process.metadata import extract_metadata
from pre_process.behavior import extract_behavior
from pre_process.fixations import extract_fixations
from pre_process.visits import extract_visits
from pre_process.lws_funnel import fixation_funnel, visit_funnel

_DEFAULT_IDENTIFICATION_ACTIONS = [
    SubjectActionTypesEnum.MARK_AND_CONFIRM,
    # SubjectActionTypesEnum.MARK_ONLY    # uncomment this to include marking-only actions
]


def read_subject(
        subject_id: int,
        exp_name: str = cnfg.EXPERIMENT_NAME,
        session: int = 1,
        data_dir: Optional[str] = None,
        verbose: bool = False,
) -> Subject:
    try:
        subj = Subject.from_pickle(exp_name=exp_name, subject_id=subject_id,)
    except FileNotFoundError:
        subj = Subject.from_raw(
            exp_name=exp_name, subject_id=subject_id, session=session, data_dir=data_dir, verbose=verbose
        )
        if "Rotem" in data_dir:     # TODO: remove this ugly hack
            subj._id = 3
        subj.to_pickle(overwrite=False)
    return subj


def process_subject(
        subj: Subject,
        identification_actions: Union[Sequence[SubjectActionTypesEnum], SubjectActionTypesEnum] = None,
        temporal_matching_threshold: float = cnfg.MAX_GAZE_TO_TRIGGER_TIME_DIFF,
        on_target_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
        visit_merging_time_threshold: float = cnfg.VISIT_MERGING_TIME_THRESHOLD,
        fixs_to_strip_threshold: int = cnfg.FIXATIONS_TO_STRIP_THRESHOLD,
        time_to_trial_end_threshold: float = cnfg.TIME_TO_TRIAL_END_THRESHOLD,
        save_fixations: bool = True,
        verbose=False,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    identification_actions = identification_actions or _DEFAULT_IDENTIFICATION_ACTIONS
    bad_actions = [
        act for act in SubjectActionTypesEnum if act not in identification_actions and act != SubjectActionTypesEnum.NO_ACTION
    ]
    targets = extract_targets(subj)
    metadata = extract_metadata(subj, bad_actions=bad_actions)
    idents = extract_behavior(
        subj, identification_actions, temporal_matching_threshold, verbose=False
    )

    fixations = extract_fixations(
        subj, identification_actions, temporal_matching_threshold, on_target_threshold_dva,
        save=save_fixations, verbose=verbose
    )
    lws_fixation_funnel = fixation_funnel(
        fixations, metadata, idents, on_target_threshold_dva, fixs_to_strip_threshold, time_to_trial_end_threshold
    )
    fixations = pd.concat([fixations, lws_fixation_funnel], axis=1)

    visits = extract_visits(fixations, on_target_threshold_dva, visit_merging_time_threshold)
    lws_visit_funnel = visit_funnel(
        visits, metadata, idents, on_target_threshold_dva, fixs_to_strip_threshold, time_to_trial_end_threshold, "min"
        # TODO: check if "weighted" distance is better
    )
    visits = pd.concat([visits, lws_visit_funnel], axis=1)
    return targets, metadata, idents, fixations, visits
