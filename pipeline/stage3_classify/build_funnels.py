from __future__ import annotations

from typing import Literal, Optional, TYPE_CHECKING

import pandas as pd

import pipeline.config as pcfg
from pipeline.stage3_classify.trial_inclusion import check_trial_inclusion_criteria
from pipeline.stage3_classify.event_classification import (
    assign_fixation_targets, check_lws_criteria, check_target_return_criteria,
)
from data_models.LWSEnums import SearchArrayCategoryEnum, ImageCategoryEnum

if TYPE_CHECKING:
    from analysis.helpers.read_data import DataStore


def build_trial_inclusion_funnel(
    data: DataStore,
    min_gaze_coverage: int | float = pcfg.DEFAULT_GAZE_COVERAGE_PERCENT_THRESHOLD,
    min_fixation_rate: float = pcfg.DEFAULT_FIXATION_RATE_THRESHOLD,
    bad_actions: Optional[pcfg.BAD_ACTIONS_TYPE] = None,
    require_actions: bool = False,
) -> pd.DataFrame:
    bad_actions = _bad_actions_as_list(bad_actions)
    trial_criteria = check_trial_inclusion_criteria(
        data.metadata, data.fixations, data.actions, data.identifications,
        min_gaze_coverage=min_gaze_coverage,
        min_fixation_rate=min_fixation_rate,
        bad_actions=bad_actions,
        require_actions=require_actions,
    )
    trial_funnel = _convert_criteria_to_funnel(trial_criteria)
    trial_funnel = (
        trial_funnel
        .reset_index(drop=False)
        .merge(
            data.metadata[["subject", "trial", "trial_category"]],
            on=["subject", "trial"],
            how="left"
        )
    )
    return _coerce_column_types(trial_funnel)


def build_event_classification_funnel(
    data: DataStore,
    funnel_type: Literal["lws", "target_return"],
    event_type: Literal["fixation", "visit"],
) -> pd.DataFrame:
    """
    Build a per-event funnel classifying each event as LWS or as a target-return.

    The cumulative chain contains only event-level criteria (from ``IS_LWS_CRITERIA`` or
    ``IS_TARGET_RETURN_CRITERIA``). Trial validity is **not** part of the funnel; consumers who need
    valid-trial-only data should filter via ``data.trial_funnel["is_valid_trial"]``.

    IMPORTANT: ``event_type`` changes the unit of analysis *and* how targets are attributed, so
    fixation-level and visit-level results are not directly comparable:

    - ``"fixation"``: one row per fixation, carrying a single ``target`` (the closest within threshold).
    - ``"visit"``: one row per (target, visit). The same fixation can contribute to visits to several
      targets, so visit counts are not fixation counts.
    """
    funnel_type = funnel_type.lower()
    event_type = event_type.lower()
    if funnel_type not in {"lws", "target_return"}:
        raise ValueError("`funnel_type` must be 'lws' or 'target_return'.")
    if event_type not in {"fixation", "visit"}:
        raise ValueError("`event_type` must be 'fixation' or 'visit'.")
    if event_type == "fixation":
        event_data = assign_fixation_targets(
            data.fixations, data.fixation_target_dists, data.on_target_threshold_dva,
        )
    else:
        event_data = data.visits
    if event_data is None or (hasattr(event_data, 'empty') and event_data.empty):
        raise ValueError(f"no {event_type} data available")
    class_criteria = _compute_event_classification_criteria(
        funnel_type=funnel_type,
        event_type=event_type,
        event_data=event_data,
        idents=data.identifications,
        on_target_threshold_dva=data.on_target_threshold_dva,
    )
    funnel_df = _convert_criteria_to_funnel(class_criteria)
    funnel_df.index = event_data.index
    out = (
        pd.concat([event_data, funnel_df], axis=1)
        .merge(
            data.metadata[["subject", "trial", "trial_category"]],
            on=["subject", "trial"],
            how="left"
        )
        .merge(
            data.targets[["subject", "trial", "target", "category", "angle"]],
            on=["subject", "trial", "target"],
            how="left"
        )
        .rename(columns={"category": "target_category", "angle": "target_angle"})
    )
    return _coerce_column_types(out)


def _bad_actions_as_list(bad_actions: Optional[pcfg.BAD_ACTIONS_TYPE]) -> list[pcfg.SubjectActionCategoryEnum]:
    if bad_actions is None:
        return list(pcfg.DEFAULT_BAD_ACTIONS)
    if isinstance(bad_actions, pcfg.SubjectActionCategoryEnum):
        return [bad_actions]
    return list(bad_actions)


def _compute_event_classification_criteria(
    funnel_type: str,
    event_type: str,
    event_data: pd.DataFrame,
    idents: pd.DataFrame,
    on_target_threshold_dva: float,
) -> pd.DataFrame:
    if funnel_type == "lws":
        return check_lws_criteria(
            event_data, idents,
            event_type=event_type,
            on_target_threshold_dva=on_target_threshold_dva,
            time_to_trial_end_threshold=pcfg.DEFAULT_MIN_MS_BEFORE_TRIAL_END,
            min_fixs_from_exemplars=pcfg.DEFAULT_MIN_FIXATIONS_FROM_STRIP,
        )
    return check_target_return_criteria(
        event_data, idents,
        event_type=event_type,
        on_target_threshold_dva=on_target_threshold_dva,
    )



def _convert_criteria_to_funnel(criteria_df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn standalone criterion columns into cumulative funnel columns, renamed so the two cannot be confused.

    Column `i` of the result is the AND of criteria `0..i`, named `upto_<criterion>` (terminal columns such as
    `is_lws` keep their name - see `funnel_config.cumulative_name`). The result is verified to be row-wise monotone:
    a True in any column guarantees True in every column before it.
    """
    funnel_df = pd.DataFrame(index=criteria_df.index)
    cumulative = pd.Series(True, index=criteria_df.index)
    for col in criteria_df.columns:
        cumulative &= criteria_df[col].fillna(False).astype(bool)
        funnel_df[pcfg.cumulative_name(col)] = cumulative
    assert_is_cumulative(funnel_df)
    return funnel_df


def assert_is_cumulative(funnel_df: pd.DataFrame, columns: Optional[list[str]] = None) -> None:
    """
    Verify the defining property of a funnel: passing step `i` implies passing every earlier step.

    Cheap enough to run on every build, and it is the one invariant that makes the cumulative column names
    trustworthy. Exported so notebooks can re-check a funnel they have filtered or merged.

    :raises ValueError: if any row passes a step without passing an earlier one.
    """
    columns = list(columns) if columns is not None else list(funnel_df.columns)
    for earlier, later in zip(columns, columns[1:]):
        violations = funnel_df[later].fillna(False) & ~funnel_df[earlier].fillna(False)
        if violations.any():
            raise ValueError(
                f"funnel is not cumulative: {int(violations.sum())} row(s) pass {later!r} but fail the earlier "
                f"{earlier!r}. First offending index: {funnel_df.index[violations][0]!r}."
            )


def _coerce_column_types(data: pd.DataFrame) -> pd.DataFrame:
    base_types = {"subject": "category", "trial": int, "target": "category", "target_angle": float}
    data = data.astype({col: typ for col, typ in base_types.items() if col in data.columns})
    # NOTE: build the categorical from the *values*, not via `from_codes` on the enum members. `from_codes` treats the
    # enum value as a position in the categories list, which only works while the enums stay zero-based and
    # contiguous; a gap or a non-zero start would either raise or silently mislabel every row.
    if "trial_category" in data.columns:
        data["trial_category"] = _as_ordered_categorical(data["trial_category"], SearchArrayCategoryEnum)
    if "target_category" in data.columns:
        data["target_category"] = _as_ordered_categorical(data["target_category"], ImageCategoryEnum)
    return data


def _as_ordered_categorical(values: pd.Series, enum_cls) -> pd.Categorical:
    """Categorical over `enum_cls`'s member names, ordered by the enum's declaration order."""
    categories = [member.name for member in enum_cls]
    unknown = set(values.dropna().unique()) - set(categories)
    if unknown:
        raise ValueError(f"values not in {enum_cls.__name__}: {sorted(unknown)}")
    return pd.Categorical(values, categories=categories, ordered=True).remove_unused_categories()
