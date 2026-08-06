import warnings
from typing import Literal, Callable

import pandas as pd

import constants as cnst
from analysis.helpers.funnels.funnel_config import IS_LWS_CRITERIA, IS_TARGET_RETURN_CRITERIA
from data_models.LWSEnums import SignalDetectionCategoryEnum

# identification categories that count as "the subject identified this target"
_HIT_CATEGORIES = frozenset({SignalDetectionCategoryEnum.HIT, SignalDetectionCategoryEnum.REPEATED_HIT})


def check_lws_criteria(
    event_data: pd.DataFrame,
    idents: pd.DataFrame,
    on_target_threshold_dva: float,
    time_to_trial_end_threshold: float,
    min_fixs_from_exemplars: int,
    event_type: Literal["fixation", "visit"],
) -> pd.DataFrame:
    """ Returns a DataFrame aligned to event_data.index with boolean columns for each criterion + `is_lws`. """
    _validate_event_type(event_type)
    ident_time = _identification_time_lookup(idents)  # Series indexed by (subject, trial, target)
    criteria_funcs: dict[str, Callable[[], pd.Series]] = {
        "on_target": lambda: is_on_target(event_data, on_target_threshold_dva, event_type),
        "before_identification": lambda: is_before_identification(event_data, ident_time),
        "not_close_to_trial_end": lambda: is_not_close_to_trial_end(event_data, time_to_trial_end_threshold),
        "not_before_exemplar_visit": lambda: is_not_before_exemplar_fixation(event_data, min_fixs_from_exemplars),
    }
    parts = [criteria_funcs[crtr]() for crtr in IS_LWS_CRITERIA]
    out = pd.concat(parts, axis=1).assign(is_lws=lambda df: df.all(axis=1)).astype(bool)
    out.index = event_data.index
    return out


def check_target_return_criteria(
    event_data: pd.DataFrame,
    idents: pd.DataFrame,
    on_target_threshold_dva: float,
    event_type: Literal["fixation", "visit"],
) -> pd.DataFrame:
    """ Returns a DataFrame aligned to event_data.index with boolean columns for each criterion + `is_target_return`. """
    _validate_event_type(event_type)
    ident_time = _identification_time_lookup(idents)
    criteria_funcs: dict[str, Callable[[], pd.Series]] = {
        "on_target": lambda: is_on_target(event_data, on_target_threshold_dva, event_type),
        "after_identification": lambda: is_after_identification(event_data, ident_time),
    }
    parts = [criteria_funcs[crtr]() for crtr in IS_TARGET_RETURN_CRITERIA]
    out = pd.concat(parts, axis=1).assign(is_target_return=lambda df: df.all(axis=1)).astype(bool)
    out.index = event_data.index
    return out


def is_on_target(
    event_data: pd.DataFrame,
    on_target_threshold_dva: float,
    event_type: Literal["fixation", "visit"],
) -> pd.Series:
    """ True if any relevant distance column <= threshold. """
    if on_target_threshold_dva <= 0:
        raise ValueError(f"`on_target_threshold_dva` must be positive, got {on_target_threshold_dva}.")
    dist_cols = _distance_columns(event_data, event_type)
    on_target = (
        event_data[dist_cols]
        .le(on_target_threshold_dva)
        .any(axis=1)
        .astype(bool)
        .rename("on_target")
    )
    return on_target


def is_before_identification(event_data: pd.DataFrame, ident_time: pd.Series) -> pd.Series:
    """
    True if the event ends before the target was identified.
    A target that was never identified has `time = inf`, so every event on it qualifies.
    Events with a missing `end_time` return False (conservative); a missing identification time raises.
    """
    t = _map_ident_time(event_data, ident_time)
    out = event_data["end_time"] < t
    out = out.fillna(False)
    return out.rename("before_identification").astype(bool)


def is_after_identification(event_data: pd.DataFrame, ident_time: pd.Series) -> pd.Series:
    """
    True if the event starts after the target was identified.
    A target that was never identified has `time = inf`, so no event on it qualifies.
    Events with a missing `start_time` return False (conservative); a missing identification time raises.
    """
    t = _map_ident_time(event_data, ident_time)
    out = event_data["start_time"] > t
    out = out.fillna(False)
    return out.rename("after_identification").astype(bool)


def is_not_close_to_trial_end(event_data: pd.DataFrame, time_to_trial_end_threshold: float) -> pd.Series:
    if time_to_trial_end_threshold < 0:
        raise ValueError(f"`time_to_trial_end_threshold` must be non-negative, got {time_to_trial_end_threshold}.")
    out = event_data["to_trial_end"] >= time_to_trial_end_threshold
    return out.rename("not_close_to_trial_end").astype(bool)


def is_not_before_exemplar_fixation(event_data: pd.DataFrame, min_fixs_from_exemplars: int) -> pd.Series:
    if min_fixs_from_exemplars < 0:
        raise ValueError(f"`min_fixs_from_exemplars` must be non-negative, got {min_fixs_from_exemplars}.")
    out = event_data["num_fixs_to_strip"] >= min_fixs_from_exemplars
    return out.rename("not_before_exemplar_visit").astype(bool)


def _validate_event_type(event_type: str) -> None:
    if event_type not in {"fixation", "visit"}:
        raise ValueError(f"Unknown event type: {event_type!r}. Expected 'fixation' or 'visit'.")


def _distance_columns(event_data: pd.DataFrame, event_type: Literal["fixation", "visit"]) -> list[str]:
    if event_type == "fixation":
        # e.g. target0_distance_dva, target1_distance_dva, ...
        dist_cols = [c for c in event_data.columns if c.startswith("target") and c.endswith("distance_dva")]
    else:
        # event_type == "visit" ->
        dist_cols = [c for c in event_data.columns if c.endswith("distance_dva")]
    if not dist_cols:
        raise ValueError(f"No distance columns found for event_type={event_type!r}.")
    return dist_cols


def _identification_time_lookup(idents: pd.DataFrame) -> pd.Series:
    """
    Build a lookup Series mapping (subject, trial, target) -> the time the target was identified.

    Only confirmed identifications count: the time is that of the *first* hit on the target, so a `repeated_hit` does
    not move it. Targets that were never identified are carried through from their `miss` row with `time = inf`, which
    makes every on-target event on them pre-identification. False alarms identify no target and carry no target label
    (see `_classify_hits_and_false_alarms`), so they cannot shadow a real hit.
    """
    required = {"subject", "trial", "target", "time", cnst.IDENTIFICATION_CATEGORY_STR}
    missing = required - set(idents.columns)
    if missing:
        raise KeyError(f"`idents` missing columns: {sorted(missing)}")
    labelled = idents[idents["target"].notna()]
    categories = labelled[cnst.IDENTIFICATION_CATEGORY_STR]
    identified = labelled[categories.isin(_HIT_CATEGORIES)]
    unidentified = labelled[categories == SignalDetectionCategoryEnum.MISS]
    keys = ["subject", "trial", "target"]
    lookup = pd.concat([
        identified.groupby(keys, observed=True)["time"].min(),    # first hit wins over any repeated hit
        unidentified.set_index(keys)["time"],                     # inf
    ])
    if lookup.index.has_duplicates:
        # a target cannot be both hit and missed; keep the hit and surface the inconsistency
        duplicated = lookup.index[lookup.index.duplicated()].tolist()
        warnings.warn(f"targets classified as both hit and miss: {duplicated}", RuntimeWarning)
        lookup = lookup[~lookup.index.duplicated(keep="first")]
    return lookup


def _map_ident_time(event_data: pd.DataFrame, id_time: pd.Series) -> pd.Series:
    """
    Map identification time into event_data rows by (subject, trial, target).
    Returns a float Series aligned to event_data.index.

    Every target has an identification time - a finite one if it was hit, `inf` if it was missed - so an unmapped key
    means the identification table and the event table disagree about which targets exist. That is a data error, and
    silently treating it as "not LWS" would hide it, so it raises.
    """
    required = {"subject", "trial", "target"}
    missing = required - set(event_data.columns)
    if missing:
        raise KeyError(f"`event_data` missing columns: {sorted(missing)}")

    key = pd.MultiIndex.from_frame(event_data[["subject", "trial", "target"]])
    mapped = pd.Series(key.map(id_time), index=event_data.index, dtype=float)
    if mapped.isna().any():
        unmapped = key[mapped.isna().to_numpy()].unique().tolist()
        raise KeyError(
            f"{len(unmapped)} (subject, trial, target) key(s) have no identification time, e.g. {unmapped[:5]}. "
            f"Every target should appear in `idents` as a hit or a miss."
        )
    return mapped
