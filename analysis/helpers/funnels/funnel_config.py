from typing import Union, Sequence

from data_models.LWSEnums import SubjectActionCategoryEnum
from config import IDENTIFICATION_ACTIONS


# Default Thresholds for Funnel Criteria
# -------------------------
DEFAULT_GAZE_COVERAGE_PERCENT_THRESHOLD = 80
DEFAULT_FIXATION_RATE_THRESHOLD = 2
BAD_ACTIONS_TYPE = Union[SubjectActionCategoryEnum, Sequence[SubjectActionCategoryEnum]]
DEFAULT_BAD_ACTIONS = tuple(
    act for act in SubjectActionCategoryEnum
    if act != SubjectActionCategoryEnum.NO_ACTION and act not in IDENTIFICATION_ACTIONS
)

DEFAULT_MIN_MS_BEFORE_TRIAL_END = 1000
DEFAULT_MIN_FIXATIONS_FROM_STRIP = 3

# Funnel Criteria Sequences
# -------------------------
TRIAL_INCLUSION_CRITERIA = [
    # sequence of criteria to determine if a trial is valid and included for further analysis
    "gaze_coverage",
    "fixation_rate",
    # "has_actions",    # uncomment to exclude trials with no subject-actions
    "no_bad_action",
    "no_miss_with_false_alarm",
]
IS_LWS_CRITERIA = [
    # sequence of criteria to determine if a fixation/visit is a LWS instance
    "on_target", "before_identification", "not_close_to_trial_end", "not_before_exemplar_visit",
]
IS_TARGET_RETURN_CRITERIA = [
    # sequence of criteria to determine if a fixation/visit is a target-return instance
    "on_target", "after_identification",
]

