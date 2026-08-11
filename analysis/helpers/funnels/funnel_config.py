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

# Funnel Column Naming
# -------------------------
# A funnel column is *cumulative*: it means "passed this criterion and every criterion before it". The standalone
# criterion of the same name - what `check_trial_inclusion_criteria` / `check_lws_criteria` return - means only
# itself. Naming them identically made the two indistinguishable in a saved CSV, so funnel columns carry a prefix.
#
# `upto_` was chosen to read as "up to and including this step". It cannot be misread as "passed only this step",
# which is the objection to a bare `passed_` prefix.
CUMULATIVE_PREFIX = "upto_"

# Columns that are conjunctions *by definition*, so cumulative and standalone coincide in meaning. They keep their
# names: `upto_is_lws` would be noise, and these are the columns downstream analyses actually select on.
TERMINAL_COLUMNS = frozenset({"is_valid_trial", "is_lws", "is_target_return"})


def cumulative_name(criterion: str) -> str:
    """Funnel column name for a criterion. Terminal columns are returned unchanged."""
    return criterion if criterion in TERMINAL_COLUMNS else f"{CUMULATIVE_PREFIX}{criterion}"


def cumulative_names(criteria: Sequence[str]) -> list:
    """Map a criteria list (e.g. IS_LWS_CRITERIA) onto the funnel's column names."""
    return [cumulative_name(crit) for crit in criteria]


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

