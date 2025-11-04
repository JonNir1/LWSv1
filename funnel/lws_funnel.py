from typing import Literal, Union, List
import pandas as pd

import config as cnfg
from funnel.helpers import run_funnel
from data_models.LWSEnums import SubjectActionCategoryEnum

_FUNNEL_STEPS = [
    # sequence of conditions to determine if a fixation/visit is a Target-Return instance
    "all",
    "trial_gaze_coverage",
    "trial_no_bad_action",
    "trial_no_false_alarm",
    "instance_on_target",
    "instance_not_outlier",
    "instance_before_identification",
    "instance_not_close_to_trial_end",
    "not_before_exemplar_visit",    # fixations/visits that precede exemplar section (bottom-strip) visits are not LWS
    "is_lws",
]


def lws_funnel(
        event_data: pd.DataFrame,
        metadata: pd.DataFrame,
        actions: pd.DataFrame,
        idents: pd.DataFrame,
        event_type: Literal["fixation", "visit"],
        bad_actions: Union[SubjectActionCategoryEnum, List[SubjectActionCategoryEnum]] = cnfg.BAD_ACTIONS,
        on_target_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
        time_to_trial_end_threshold: float = cnfg.TIME_TO_TRIAL_END_THRESHOLD,
        exemplar_visit_threshold: int = cnfg.FIXATIONS_TO_STRIP_THRESHOLD,
        verbose: bool = False,
) -> pd.DataFrame:
    """ Run the LWS funnel analysis on the provided events' data. """
    funnel_results = run_funnel(
        event_data=event_data,
        metadata=metadata,
        actions=actions,
        idents=idents,
        event_type=event_type,
        bad_actions=bad_actions,
        on_target_threshold_dva=on_target_threshold_dva,
        time_to_trial_end_threshold=time_to_trial_end_threshold,
        exemplar_visit_threshold=exemplar_visit_threshold,
        verbose=verbose,
        steps=_FUNNEL_STEPS,
    )
    funnel_results["is_lws"] = funnel_results.apply(lambda row: row.all(), axis=1)
    return funnel_results
