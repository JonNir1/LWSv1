import os
import warnings
from dataclasses import dataclass
from typing import Optional, Literal

import pandas as pd
from numpy import isnan


@dataclass(frozen=True)
class LoadedData:
    targets: Optional[pd.DataFrame]
    actions: Optional[pd.DataFrame]
    metadata: Optional[pd.DataFrame]
    identifications: Optional[pd.DataFrame]
    fixations: Optional[pd.DataFrame]
    visits: Optional[pd.DataFrame]


def read_data(
        dir_path: str,
        drop_bad_eye: bool = True,
        drop_outliers: bool = True,
        missing: Literal["warn", "ignore", "raise"] = "warn",
) -> LoadedData:
    """
    Read analysis inputs from a directory of pickle files.
    Returns a LoadedData object with fields for targets, actions, metadata, identifications, fixations, and visits.
    If a file is missing, the corresponding field will be set to None, and the behavior depends on the `missing` argument.
    """
    targets = _load(dir_path, "targets", missing)
    actions = _load(dir_path, "actions", missing)
    metadata = _load(dir_path, "metadata", missing)
    idents = _load(dir_path, "idents", missing)
    fixations = _load(dir_path, "fixations", missing)
    visits = _load(dir_path, "visits", missing)
    if drop_bad_eye and metadata is not None:
        if fixations is not None:
            fixations = _drop_bad_eye(fixations, metadata)
        if visits is not None:
            visits = _drop_bad_eye(visits, metadata)
    if drop_outliers and fixations is not None:
        fixations = fixations.loc[fixations["outlier_reasons"].map(
                # treat None as not-outlier:
                lambda rsns: (isinstance(rsns, list) and len(rsns) == 0) or rsns is None
        )]
    return LoadedData(
        targets=targets,
        actions=actions,
        metadata=metadata,
        identifications=idents,
        fixations=fixations,
        visits=visits,
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
    """
    Attempts to load a DataFrame from a pickle file in the specified directory.
    If the file is not found, behavior depends on the `missing` parameter.
    """
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
    """ Drop events from the non-dominant eye based on the dominant eye information in the metadata. """
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
