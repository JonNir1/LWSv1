import os
import warnings
from dataclasses import dataclass
from typing import Optional, Literal

import pandas as pd
from numpy import isnan

import config as cnfg
import pipeline.config as pcfg
from pipeline.stage2_align.fixations_to_targets import fixations_to_targets
from pipeline.stage2_align.build_visits import build_visits
from pipeline.stage2_align.target_identifications import build_identifications
from pipeline.stage3_classify.run_stage3 import run_stage3


FIXATION_EVENT_TYPE = "FIXATION"

_DATA_CACHE: dict[tuple, "DataStore"] = {}


@dataclass(frozen=True)
class DataStore:
    """All tables needed for analysis: stage-1 pickles plus stage-2 and stage-3 on-the-fly outputs."""

    # stage 1 (persisted)
    icons: Optional[pd.DataFrame]
    actions: Optional[pd.DataFrame]
    metadata: Optional[pd.DataFrame]
    eye_movements: Optional[pd.DataFrame]

    # stage 2 (computed on-the-fly)
    fixation_target_dists: pd.DataFrame
    visits: pd.DataFrame
    identifications: pd.DataFrame

    # stage 3 (computed on-the-fly)
    trial_funnel: pd.DataFrame
    event_funnels: dict[str, pd.DataFrame]

    # thresholds used to produce stage-2 outputs
    on_target_threshold_dva: float
    visit_merging_time_threshold: float

    # thresholds used to produce stage-3 outputs
    min_gaze_coverage: float
    min_fixation_rate: float

    @property
    def fixations(self) -> Optional[pd.DataFrame]:
        if self.eye_movements is None:
            return None
        return self.eye_movements.loc[self.eye_movements["event_type"] == FIXATION_EVENT_TYPE]

    @property
    def targets(self) -> Optional[pd.DataFrame]:
        if self.icons is None:
            return None
        targets = self.icons.loc[self.icons["is_target"]].drop(columns=["is_target"])
        return targets.rename(columns={"icon": "target"}).reset_index(drop=True)


def load_data(
        dir_path: str,
        drop_bad_eye: bool = True,
        drop_outliers: bool = True,
        missing: Literal["warn", "ignore", "raise"] = "warn",
        on_target_threshold_dva: float = pcfg.ON_TARGET_THRESHOLD_DVA,
        visit_merging_time_threshold: float = pcfg.VISIT_MERGING_TIME_THRESHOLD,
        identification_actions=None,
        min_gaze_coverage: float = pcfg.DEFAULT_GAZE_COVERAGE_PERCENT_THRESHOLD,
        min_fixation_rate: float = pcfg.DEFAULT_FIXATION_RATE_THRESHOLD,
) -> DataStore:
    """Load stage-1 pickles and compute stage-2 alignment and stage-3 classification."""
    if identification_actions is None:
        identification_actions = pcfg.IDENTIFICATION_ACTIONS

    icons = _load(dir_path, "icons", missing)
    actions = _load(dir_path, "actions", missing)
    metadata = _load(dir_path, "metadata", missing)
    eye_movements = _load(dir_path, "eye_movements", missing)
    if drop_bad_eye and metadata is not None:
        if eye_movements is not None:
            eye_movements = _drop_bad_eye(eye_movements, metadata)
    if drop_outliers and eye_movements is not None:
        eye_movements = eye_movements.loc[~eye_movements["is_outlier"].fillna(False).astype(bool)]

    fixations = None
    if eye_movements is not None:
        fixations = eye_movements.loc[eye_movements["event_type"] == FIXATION_EVENT_TYPE]

    dists = fixations_to_targets(fixations, icons, metadata)
    visits = build_visits(fixations, dists, on_target_threshold_dva, visit_merging_time_threshold)
    idents = build_identifications(
        fixations, icons, actions, metadata,
        identification_actions, on_target_threshold_dva,
    )

    # build a partial DataStore (without stage 3) so run_stage3 can use it
    partial = DataStore(
        icons=icons,
        actions=actions,
        metadata=metadata,
        eye_movements=eye_movements,
        fixation_target_dists=dists,
        visits=visits,
        identifications=idents,
        trial_funnel=pd.DataFrame(),
        event_funnels={},
        on_target_threshold_dva=on_target_threshold_dva,
        visit_merging_time_threshold=visit_merging_time_threshold,
        min_gaze_coverage=min_gaze_coverage,
        min_fixation_rate=min_fixation_rate,
    )

    trial_funnel, event_funnels = run_stage3(
        partial,
        min_gaze_coverage=min_gaze_coverage,
        min_fixation_rate=min_fixation_rate,
    )

    return DataStore(
        icons=icons,
        actions=actions,
        metadata=metadata,
        eye_movements=eye_movements,
        fixation_target_dists=dists,
        visits=visits,
        identifications=idents,
        trial_funnel=trial_funnel,
        event_funnels=event_funnels,
        on_target_threshold_dva=on_target_threshold_dva,
        visit_merging_time_threshold=visit_merging_time_threshold,
        min_gaze_coverage=min_gaze_coverage,
        min_fixation_rate=min_fixation_rate,
    )


def load_analysis_data(
        dir_path: str = None,
        funnel_type: str = "lws",
        event_type: str = "visit",
        drop_bad_eye: bool = True,
        drop_outliers: bool = True,
        **load_kwargs,
) -> tuple["DataStore", pd.DataFrame]:
    """
    Convenience wrapper: loads data and retrieves a pre-built event funnel.

    Returns (DataStore, funnel_df). The funnel contains all events (including those from invalid trials).
    To restrict to valid trials, filter explicitly::

        valid = data.trial_funnel.loc[data.trial_funnel["is_valid_trial"], ["subject", "trial"]]
        funnel_df = funnel_df.merge(valid, on=["subject", "trial"])
    """
    if dir_path is None:
        dir_path = cnfg.OUTPUT_PATH

    cache_key = (
        dir_path, drop_bad_eye, drop_outliers,
        load_kwargs.get("on_target_threshold_dva", pcfg.ON_TARGET_THRESHOLD_DVA),
        load_kwargs.get("visit_merging_time_threshold", pcfg.VISIT_MERGING_TIME_THRESHOLD),
        load_kwargs.get("min_gaze_coverage", pcfg.DEFAULT_GAZE_COVERAGE_PERCENT_THRESHOLD),
        load_kwargs.get("min_fixation_rate", pcfg.DEFAULT_FIXATION_RATE_THRESHOLD),
    )
    if cache_key in _DATA_CACHE:
        data = _DATA_CACHE[cache_key]
    else:
        data = load_data(dir_path, drop_bad_eye=drop_bad_eye, drop_outliers=drop_outliers, **load_kwargs)
        _DATA_CACHE[cache_key] = data

    funnel_key = f"{funnel_type}_{event_type}"
    if funnel_key not in data.event_funnels:
        raise KeyError(
            f"No event funnel '{funnel_key}'. "
            f"Available: {sorted(data.event_funnels.keys())}"
        )
    return data, data.event_funnels[funnel_key]


def parse_as_categorical(series: pd.Series, enum_cls, ordered: bool) -> pd.Categorical:
    mapped = series.map(lambda val: val if val == "all" else enum_cls[val].name)
    cat = pd.Categorical(
        mapped,
        categories=[e.name for e in enum_cls] + ["all"],
        ordered=ordered,
    )
    return cat.remove_unused_categories()


def _load(dir_path: str, name: str, missing: Literal["warn", "ignore", "raise"]) -> Optional[pd.DataFrame]:
    path = os.path.join(dir_path, f"{name}.pkl")
    try:
        return pd.read_pickle(path)
    except FileNotFoundError:
        missing = missing.lower()
        if missing not in {"warn", "ignore", "raise"}:
            raise ValueError("Argument `missing` must be one of {'warn','ignore','raise'}.")
        msg = f"{name}.pkl not found in {dir_path!r}"
        if missing == "raise":
            raise FileNotFoundError(msg)
        if missing == "warn":
            warnings.warn(msg)
        return None


def _drop_bad_eye(events: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    required_event_cols = {"subject", "trial", "eye"}
    missing_cols = required_event_cols - set(events.columns)
    if missing_cols:
        raise KeyError(f"events is missing columns: {sorted(missing_cols)}")
    required_meta_cols = {"subject", "trial", "dominant_eye"}
    missing_cols = required_meta_cols - set(metadata.columns)
    if missing_cols:
        raise KeyError(f"metadata is missing columns: {sorted(missing_cols)}")
    metadata_dom = metadata.set_index(["subject", "trial"])["dominant_eye"]
    events_dom = events.set_index(["subject", "trial"]).index.map(metadata_dom)
    out = events.loc[events["eye"].to_numpy() == events_dom.to_numpy()]
    return out
