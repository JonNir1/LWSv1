"""
All pipeline configuration: detection thresholds, alignment parameters, classification criteria, and naming helpers.

Organized by stage:
  Stage 1 (parse): event-detection duration bounds
  Stage 2 (align): on-target threshold, visit merging, identification actions
  Stage 3 (classify): trial-inclusion and event-classification criteria, funnel naming
"""

from typing import Union, Sequence

from data_models.LWSEnums import SubjectActionCategoryEnum


# Stage 1: Eye-Movement Detection
# --------------------------------
# Duration bounds (ms) for `peyes` event detection. Events outside these bounds are flagged as outliers
# and dropped by `load_data(drop_outliers=True)`.
MIN_EVENT_DURATION_MS = 5
FIXATION_MIN_DURATION_MS = 50
# Measured 2026-08-06 over 116,947 fixations: the right tail decays smoothly and monotonically with no
# secondary mode (p99 = 770 ms, p99.9 = 1468 ms, max = 2797 ms). There is no empirical bump to cut at,
# so the peyes default is kept rather than replaced by an arbitrary cut.
# TODO(T3): check the visual-search literature for a principled upper bound on fixation duration.
FIXATION_MAX_DURATION_MS = 2500
SACCADE_MIN_DURATION_MS = MIN_EVENT_DURATION_MS
SACCADE_MAX_DURATION_MS = 200


# Stage 2: Alignment
# --------------------------------
ON_TARGET_THRESHOLD_DVA = 1.75
VISIT_MERGING_TIME_THRESHOLD = 100.0
IDENTIFICATION_ACTIONS = [
    SubjectActionCategoryEnum.MARK_AND_CONFIRM,
    # SubjectActionCategoryEnum.MARK_ONLY    # uncomment to include marking-only actions
]


# Stage 3: Classification
# --------------------------------

# Trial-inclusion thresholds
DEFAULT_GAZE_COVERAGE_PERCENT_THRESHOLD = 80
DEFAULT_FIXATION_RATE_THRESHOLD = 2

# Event-classification thresholds
DEFAULT_MIN_MS_BEFORE_TRIAL_END = 1000
DEFAULT_MIN_FIXATIONS_FROM_STRIP = 3

# Bad actions: any subject action that is not NO_ACTION and not an identification action
BAD_ACTIONS_TYPE = Union[SubjectActionCategoryEnum, Sequence[SubjectActionCategoryEnum]]
DEFAULT_BAD_ACTIONS = tuple(
    act for act in SubjectActionCategoryEnum
    if act != SubjectActionCategoryEnum.NO_ACTION and act not in IDENTIFICATION_ACTIONS
)

# Criteria lists (order matters: each criterion is AND-ed cumulatively)
TRIAL_INCLUSION_CRITERIA = [
    "gaze_coverage",
    "fixation_rate",
    # "has_actions",    # uncomment to exclude trials with no subject-actions
    "no_bad_action",
    "no_miss_with_false_alarm",
]
IS_LWS_CRITERIA = [
    "on_target", "before_identification", "not_close_to_trial_end", "not_before_exemplar_visit",
]
IS_TARGET_RETURN_CRITERIA = [
    "on_target", "after_identification",
]


# Funnel Column Naming
# --------------------------------
# A funnel column is *cumulative*: it means "passed this criterion and every criterion before it".
# `upto_` reads as "up to and including this step".
CUMULATIVE_PREFIX = "upto_"

# Terminal columns are conjunctions by definition, so cumulative and standalone coincide. They keep
# their names (e.g. `is_lws` rather than `upto_is_lws`).
TERMINAL_COLUMNS = frozenset({"is_valid_trial", "is_lws", "is_target_return"})


def cumulative_name(criterion: str) -> str:
    """Funnel column name for a criterion. Terminal columns are returned unchanged."""
    if criterion in TERMINAL_COLUMNS:
        return criterion
    return f"{CUMULATIVE_PREFIX}{criterion}"


def cumulative_names(criteria: Sequence[str]) -> list[str]:
    """Map a criteria list onto the funnel's column names."""
    return [cumulative_name(crit) for crit in criteria]
