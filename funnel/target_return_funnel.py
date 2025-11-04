from typing import Literal, Union, List
import pandas as pd

import config as cnfg
from funnel.helpers import run_funnel
from data_models.LWSEnums import SubjectActionCategoryEnum

_FUNNEL_STEPS = [
    # sequence of conditions to determine if a fixation/visit is a LWS instance
    "all",
    "trial_gaze_coverage",
    "trial_no_bad_action",
    "trial_no_false_alarm",
    "instance_on_target",
    "instance_not_outlier",
    "instance_after_identification",
    "is_target_return"
]


def target_return_funnel(
        event_data: pd.DataFrame,
        metadata: pd.DataFrame,
        actions: pd.DataFrame,
        idents: pd.DataFrame,
        event_type: Literal["fixation", "visit"],
        bad_actions: Union[SubjectActionCategoryEnum, List[SubjectActionCategoryEnum]] = cnfg.BAD_ACTIONS,
        on_target_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
) -> pd.DataFrame:
    """ Run the Target-Return funnel analysis on the provided events' data. """
    funnel_results = run_funnel(
        event_data=event_data,
        metadata=metadata,
        actions=actions,
        idents=idents,
        event_type=event_type,
        bad_actions=bad_actions,
        on_target_threshold_dva=on_target_threshold_dva,
        steps=_FUNNEL_STEPS,
        time_to_trial_end_threshold=0,      # not used in target-return funnel
        exemplar_visit_threshold=0,         # not used in target-return funnel
    )
    funnel_results["is_target_return"] = funnel_results.apply(lambda row: row.all(), axis=1)
    return funnel_results
