from typing import Literal, Optional

import pandas as pd

import config as cnfg
import analysis.helpers.funnels.funnel_config as fcfg
from analysis.helpers.read_data import read_data
from analysis.helpers.funnels.trial_inclusion import check_trial_inclusion_criteria
from analysis.helpers.funnels.event_classification import check_lws_criteria, check_target_return_criteria
from data_models.LWSEnums import SearchArrayCategoryEnum, ImageCategoryEnum


def build_trial_inclusion_funnel(
    data_dir: str,
    min_gaze_coverage: int | float = fcfg.DEFAULT_GAZE_COVERAGE_PERCENT_THRESHOLD,
    min_fixation_rate: float = fcfg.DEFAULT_FIXATION_RATE_THRESHOLD,
    bad_actions: Optional[fcfg.BAD_ACTIONS_TYPE] = None,
    require_actions: bool = False,
) -> pd.DataFrame:
    bad_actions = _bad_actions_as_list(bad_actions)
    loaded = read_data(data_dir, drop_bad_eye=True)
    trial_criteria = check_trial_inclusion_criteria(
        loaded.metadata, loaded.fixations, loaded.actions, loaded.identifications,
        min_gaze_coverage=min_gaze_coverage,
        min_fixation_rate=min_fixation_rate,
        bad_actions=bad_actions,
        require_actions=require_actions,
    )
    trial_funnel = _convert_criteria_to_funnel(trial_criteria)
    trial_funnel = (    # attach trial category from metadata
        trial_funnel
        .reset_index(drop=False)
        .merge(
            loaded.metadata[["subject", "trial", "trial_category"]],
            on=["subject", "trial"],
            how="left"
        )
    )
    return _coerce_column_types(trial_funnel)


def build_event_classification_funnel(
    data_dir: str,
    funnel_type: Literal["lws", "target_return"],
    event_type: Literal["fixation", "visit"],
    min_gaze_coverage: int | float = fcfg.DEFAULT_GAZE_COVERAGE_PERCENT_THRESHOLD,
    min_fixation_rate: float = fcfg.DEFAULT_FIXATION_RATE_THRESHOLD,
    bad_actions: Optional[fcfg.BAD_ACTIONS_TYPE] = None,
    require_actions: bool = False,
    on_target_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
    exclude: Literal["none", "invalid_trials", "outliers", "both"] = "both",
) -> pd.DataFrame:
    funnel_type = funnel_type.lower()
    event_type = event_type.lower()
    exclude = exclude.lower()
    if funnel_type not in {"lws", "target_return"}:
        raise ValueError("`funnel_type` must be 'lws' or 'target_return'.")
    if event_type not in {"fixation", "visit"}:
        raise ValueError("`event_type` must be 'fixation' or 'visit'.")
    if exclude not in {"none", "invalid_trials", "outliers", "both"}:
        raise ValueError("`exclude` must be 'none', 'invalid_trials', 'outliers', or 'both'.")
    if event_type == "visit" and exclude in {"outliers", "both"}:
        # `drop_outliers` filters fixations on their `outlier_reasons`; the visit table carries no such column, so the
        # request would silently do nothing. What makes a *visit* an outlier is undecided (CODE_REVIEW T2), so refuse
        # rather than invent a rule.
        raise NotImplementedError(
            f"outlier exclusion is not implemented for visits (exclude={exclude!r}). Visits carry no outlier "
            f"information, so the request would be silently ignored. Use exclude='invalid_trials' or 'none', or "
            f"decide the visit-outlier rule first (see CODE_REVIEW.md T2 / H2)."
        )
    bad_actions = _bad_actions_as_list(bad_actions)
    loaded = read_data(data_dir, drop_bad_eye=True, drop_outliers=exclude in {"outliers", "both"})
    event_data = loaded.fixations if event_type == "fixation" else loaded.visits
    trial_criteria = check_trial_inclusion_criteria(
        loaded.metadata, loaded.fixations, loaded.actions, loaded.identifications,
        min_gaze_coverage=min_gaze_coverage,
        min_fixation_rate=min_fixation_rate,
        bad_actions=bad_actions,
        require_actions=require_actions,
    )
    class_criteria = _compute_event_classification_criteria(
        funnel_type=funnel_type,
        event_type=event_type,
        event_data=event_data,
        idents=loaded.identifications,
        on_target_threshold_dva=on_target_threshold_dva,
    )
    # build a joint funnel table aligned to event_data rows
    joint_criteria = _join_trial_and_event_criteria(
        event_data, class_criteria, trial_criteria,
    )
    funnel_df = _convert_criteria_to_funnel(joint_criteria)
    funnel_df.index = event_data.index
    # enrich event data with funnel columns + trial and target metadata
    out = (
        pd.concat([event_data, funnel_df], axis=1)
        .merge(
            loaded.metadata[["subject", "trial", "trial_category"]],
            on=["subject", "trial"],
            how="left"
        )
        .merge(
            loaded.targets[["subject", "trial", "target", "category", "angle"]],
            on=["subject", "trial", "target"],
            how="left"
        )
        .rename(columns={"category": "target_category", "angle": "target_angle"})
    )
    # drop invalid trials if specified
    if exclude in {"invalid_trials", "both"}:
        if "is_valid_trial" not in out.columns:
            raise KeyError("Cannot exclude invalid trials because 'is_valid_trial' column is missing.")
        out = out[out["is_valid_trial"].fillna(False)]
    return _coerce_column_types(out)


def _bad_actions_as_list(bad_actions: Optional[fcfg.BAD_ACTIONS_TYPE]) -> list[fcfg.SubjectActionCategoryEnum]:
    if bad_actions is None:
        return list(fcfg.DEFAULT_BAD_ACTIONS)
    if isinstance(bad_actions, fcfg.SubjectActionCategoryEnum):
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
            time_to_trial_end_threshold=fcfg.DEFAULT_MIN_MS_BEFORE_TRIAL_END,
            min_fixs_from_exemplars=fcfg.DEFAULT_MIN_FIXATIONS_FROM_STRIP,
        )
    return check_target_return_criteria(
        event_data, idents,
        event_type=event_type,
        on_target_threshold_dva=on_target_threshold_dva,
    )


def _join_trial_and_event_criteria(
        event_data: pd.DataFrame,
        event_criteria: pd.DataFrame,
        trial_criteria: pd.DataFrame,
) -> pd.DataFrame:
    """ Return boolean criteria columns aligned to event_data rows, by bringing in trial_criteria on (subject, trial). """
    col_order = list(trial_criteria.columns) + list(event_criteria.columns)
    event_with_crit = pd.concat(
        [event_data[["subject", "trial"]], event_criteria],
        axis=1
    )
    joined = (
        event_with_crit
        .merge(
            trial_criteria.reset_index(inplace=False), on=["subject", "trial"], how="left"
        )
        .drop(columns=["subject", "trial"])
        .loc[:, col_order]
    )
    return joined


def _convert_criteria_to_funnel(criteria_df: pd.DataFrame) -> pd.DataFrame:
    funnel_df = pd.DataFrame(index=criteria_df.index)
    cumulative = pd.Series(True, index=criteria_df.index)
    for col in criteria_df.columns:
        cumulative &= criteria_df[col].fillna(False).astype(bool)
        funnel_df[col] = cumulative
    return funnel_df


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
