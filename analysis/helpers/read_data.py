import os
import warnings
from dataclasses import dataclass
from typing import Optional, Literal

import pandas as pd
from numpy import isnan


FIXATION_EVENT_TYPE = "FIXATION"


@dataclass(frozen=True)
class LoadedData:
    icons: Optional[pd.DataFrame]
    actions: Optional[pd.DataFrame]
    metadata: Optional[pd.DataFrame]
    identifications: Optional[pd.DataFrame]
    eye_movements: Optional[pd.DataFrame]
    visits: Optional[pd.DataFrame]

    @property
    def fixations(self) -> Optional[pd.DataFrame]:
        """
        The fixation subset of `eye_movements`.

        Derived rather than loaded: `fixations.pkl` was retired in favour of the full event table, which is its
        superset - saccades and blinks were being detected and then discarded. Row-for-row and column-for-column
        this is what `fixations.pkl` used to hold, `event` values included (`event` has always been the rank among
        *all* events of an eye, so keeping the saccades only closes the gaps).
        """
        if self.eye_movements is None:
            return None
        return self.eye_movements.loc[self.eye_movements["event_type"] == FIXATION_EVENT_TYPE]

    @property
    def targets(self) -> Optional[pd.DataFrame]:
        """
        The target subset of `icons`, with the identifier column renamed to `target`.

        Derived rather than loaded: `targets.pkl` was retired in favour of the full icon table, since a target's
        identity is just an icon that happens to be a target. Existing consumers keep working unchanged.
        """
        if self.icons is None:
            return None
        targets = self.icons.loc[self.icons["is_target"]].drop(columns=["is_target"])
        return targets.rename(columns={"icon": "target"}).reset_index(drop=True)


def read_data(
        dir_path: str,
        drop_bad_eye: bool = True,
        drop_outliers: bool = True,
        missing: Literal["warn", "ignore", "raise"] = "warn",
) -> LoadedData:
    """
    Read analysis inputs from a directory of pickle files.
    Returns a LoadedData object with fields for icons, actions, metadata, identifications and eye movements, plus
    the derived `targets` and `fixations` views.
    If a file is missing, the corresponding field will be set to None, and the behavior depends on the `missing` argument.
    """
    icons = _load(dir_path, "icons", missing)
    actions = _load(dir_path, "actions", missing)
    metadata = _load(dir_path, "metadata", missing)
    idents = _load(dir_path, "idents", missing)
    eye_movements = _load(dir_path, "eye_movements", missing)
    visits = _load(dir_path, "visits", "ignore")    # not produced at present; see `Subject.get_visits`
    if drop_bad_eye and metadata is not None:
        if eye_movements is not None:
            eye_movements = _drop_bad_eye(eye_movements, metadata)
        if visits is not None:
            visits = _drop_bad_eye(visits, metadata)
    if drop_outliers and eye_movements is not None:
        # `is_outlier` is exactly `len(outlier_reasons) > 0`, but is a bool column rather than a per-row `map` over
        # object cells. Note this now drops outlier *saccades and blinks* as well, not only fixations.
        eye_movements = eye_movements.loc[~eye_movements["is_outlier"].fillna(False).astype(bool)]
    if drop_outliers and visits is not None:
        visits = _drop_outlier_visits(visits)
    return LoadedData(
        icons=icons,
        actions=actions,
        metadata=metadata,
        identifications=idents,
        eye_movements=eye_movements,
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


def _drop_outlier_visits(visits: pd.DataFrame) -> pd.DataFrame:
    """
    Drop visits whose fixations are *all* outliers.

    A visit is an episode of looking at a target, built from several fixations. One outlier fixation inside it does
    not invalidate the episode - and on the current build 99.5% of outliers are sub-50 ms blips, so an "any outlier"
    rule would discard 12.4% of visits over what is mostly detector noise. Requiring *all* fixations to be outliers
    drops 1.5% and leaves only visits with no trustworthy sample behind them.

    Visits built before this column existed are kept, with a warning: their outlier composition is unknown, and
    silently treating them as clean would hide it.
    """
    required = {"num_fixations", "num_outlier_fixations"}
    if not required.issubset(visits.columns):
        warnings.warn(
            "visits table predates outlier tracking (no num_outlier_fixations column), so outlier visits cannot be "
            "dropped; re-run the pipeline to rebuild it.",
            RuntimeWarning,
        )
        return visits
    all_outliers = visits["num_outlier_fixations"] >= visits["num_fixations"]
    return visits.loc[~all_outliers]


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
