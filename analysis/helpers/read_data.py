import os
import warnings
from dataclasses import dataclass
from typing import Optional, Literal

import pandas as pd
from numpy import isnan

import config as cnfg
from pipeline.align.fixations_to_targets import fixations_to_targets
from pipeline.align.build_visits import build_visits
from pipeline.align.target_identifications import build_identifications


FIXATION_EVENT_TYPE = "FIXATION"


@dataclass(frozen=True)
class LoadedData:
    icons: Optional[pd.DataFrame]
    actions: Optional[pd.DataFrame]
    metadata: Optional[pd.DataFrame]
    eye_movements: Optional[pd.DataFrame]

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


@dataclass(frozen=True)
class AlignedData:
    loaded: LoadedData
    fixation_target_dists: pd.DataFrame
    visits: pd.DataFrame
    identifications: pd.DataFrame
    on_target_threshold_dva: float
    visit_merging_time_threshold: float

    @property
    def fixations(self) -> Optional[pd.DataFrame]:
        return self.loaded.fixations

    @property
    def targets(self) -> Optional[pd.DataFrame]:
        return self.loaded.targets

    @property
    def icons(self) -> Optional[pd.DataFrame]:
        return self.loaded.icons

    @property
    def actions(self) -> Optional[pd.DataFrame]:
        return self.loaded.actions

    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        return self.loaded.metadata

    @property
    def eye_movements(self) -> Optional[pd.DataFrame]:
        return self.loaded.eye_movements


def read_data(
        dir_path: str,
        drop_bad_eye: bool = True,
        drop_outliers: bool = True,
        missing: Literal["warn", "ignore", "raise"] = "warn",
) -> LoadedData:
    """Read stage-1 outputs from a directory of pickle files."""
    icons = _load(dir_path, "icons", missing)
    actions = _load(dir_path, "actions", missing)
    metadata = _load(dir_path, "metadata", missing)
    eye_movements = _load(dir_path, "eye_movements", missing)
    if drop_bad_eye and metadata is not None:
        if eye_movements is not None:
            eye_movements = _drop_bad_eye(eye_movements, metadata)
    if drop_outliers and eye_movements is not None:
        eye_movements = eye_movements.loc[~eye_movements["is_outlier"].fillna(False).astype(bool)]
    return LoadedData(
        icons=icons,
        actions=actions,
        metadata=metadata,
        eye_movements=eye_movements,
    )


def align_data(
        loaded: LoadedData,
        on_target_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
        visit_merging_time_threshold: float = 100.0,
        identification_actions=None,
) -> AlignedData:
    """Compute stage-2 alignment from stage-1 outputs.

    :param loaded: stage-1 data from read_data().
    :param on_target_threshold_dva: distance threshold for on-target classification.
    :param visit_merging_time_threshold: max temporal gap (ms) for merging consecutive on-target fixations.
    :param identification_actions: action categories that count as identification attempts.
    """
    if identification_actions is None:
        identification_actions = cnfg.IDENTIFICATION_ACTIONS

    dists = fixations_to_targets(loaded.fixations, loaded.icons, loaded.metadata)
    visits = build_visits(loaded.fixations, dists, on_target_threshold_dva, visit_merging_time_threshold)
    idents = build_identifications(
        loaded.fixations, loaded.icons, loaded.actions, loaded.metadata,
        identification_actions, on_target_threshold_dva,
    )
    return AlignedData(
        loaded=loaded,
        fixation_target_dists=dists,
        visits=visits,
        identifications=idents,
        on_target_threshold_dva=on_target_threshold_dva,
        visit_merging_time_threshold=visit_merging_time_threshold,
    )


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
