"""Eye-movement detection and tabulation.

Detection: `detect_eye_movements` runs the Engbert algorithm via `peyes` on raw gaze samples, producing event
labels and (optionally) event objects per eye.

Tabulation: `process_trial_events` reshapes the detected events into the persisted event table, adding derived
columns (`to_trial_end`, split `x`/`y`, `num_fixs_to_strip`). Target distances are computed in stage 2 by
`pipeline.stage2_align.fixations_to_targets`.
"""

from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd
import peyes

import config as cnfg
import constants as cnst
from data_models.LWSEnums import DominantEyeEnum
from data_models.SearchArray import SearchArray


# ---------------------------------------------------------------------------
# peyes configuration and detector
# ---------------------------------------------------------------------------

def configure_peyes() -> None:
    """
    Push this project's screen geometry and event-duration bounds into `peyes`' module-level configuration.

    `peyes` keeps these as global state. Left unset it falls back to its own defaults, which are *a* Tobii rig but not
    this one (53.1 cm wide vs `TOBII_MONITOR`'s 53.0), and to duration bounds never chosen for this paradigm. Both
    feed `get_outlier_reasons`, and therefore decide which fixations survive `read_data(drop_outliers=True)` - so
    they are set explicitly here rather than inherited.

    Idempotent; safe to call more than once.
    """
    peyes.set_screen_monitor(
        width_cm=cnst.TOBII_MONITOR.width_mm / 10,
        height_cm=cnst.TOBII_MONITOR.height_mm / 10,
        width_px=cnst.TOBII_MONITOR.width,
        height_px=cnst.TOBII_MONITOR.height,
    )
    peyes.set_event_configurations(
        "fixation", min_duration=cnfg.FIXATION_MIN_DURATION_MS, max_duration=cnfg.FIXATION_MAX_DURATION_MS,
    )
    peyes.set_event_configurations(
        "saccade", min_duration=cnfg.SACCADE_MIN_DURATION_MS, max_duration=cnfg.SACCADE_MAX_DURATION_MS,
    )


## Eye-Movement Detection Configurations ##
# NOTE: applied at import because `_DETECTOR` below is built at import. `run_pipeline` calls `configure_peyes()`
# again explicitly, so the configuration is visible at the entry point rather than only as an import side effect.
configure_peyes()
_DETECTOR = peyes.create_detector(
    algorithm="Engbert",
    missing_value=cnst.MISSING_VALUE,
    min_event_duration=cnfg.MIN_EVENT_DURATION_MS,
    pad_blinks_time=0,      # ms
)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_eye_movements(
        gaze: pd.DataFrame,
        eye: DominantEyeEnum,
        viewer_distance_cm: float,
        detector=_DETECTOR,
        pixel_size_cm: float = cnst.PIXEL_SIZE_CM,
        only_labels: bool = True,
) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    t = gaze[cnst.TIME_STR].values
    x = gaze[cnst.RIGHT_X_STR if eye == DominantEyeEnum.RIGHT else cnst.LEFT_X_STR].values
    y = gaze[cnst.RIGHT_Y_STR if eye == DominantEyeEnum.RIGHT else cnst.LEFT_Y_STR].values
    labels, _ = detector.detect(t=t, x=x, y=y, viewer_distance_cm=viewer_distance_cm, pixel_size_cm=pixel_size_cm)
    labels = pd.Series(
        labels,
        index=gaze.index,
        name=cnst.RIGHT_LABEL_STR if eye == DominantEyeEnum.RIGHT else cnst.LEFT_LABEL_STR
    )
    if only_labels:
        return labels
    pupil = gaze[cnst.RIGHT_PUPIL_STR if eye == DominantEyeEnum.RIGHT else cnst.LEFT_PUPIL_STR].values
    events = peyes.create_events(
        labels=labels, t=t, x=x, y=y, pupil=pupil,
        viewer_distance=viewer_distance_cm, pixel_size=pixel_size_cm
    )
    events = pd.Series(events, name=cnst.RIGHT_EVENT_STR if eye == DominantEyeEnum.RIGHT else cnst.LEFT_EVENT_STR)
    return labels, events


# ---------------------------------------------------------------------------
# Tabulation
# ---------------------------------------------------------------------------

_FIXATION_LABEL = peyes.parse_label(cnst.FIXATION_STR)
_EVENT_TYPE_STR = f"{cnst.EVENT_STR}_type"
_FIXATION_ONLY_COLUMNS = [cnst.X, cnst.Y, "num_fixs_to_strip"]

_COLUMN_ORDER = [
    cnst.EYE_STR, cnst.EVENT_STR, _EVENT_TYPE_STR,
    cnst.START_TIME_STR, cnst.END_TIME_STR, "duration", "to_trial_end",
    cnst.X, cnst.Y, "start_x", "start_y", "end_x", "end_y",
    "std_x", "std_y", "dispersion", "ellipse_area",
    cnst.DISTANCE_STR, "amplitude", "azimuth", "cumulative_distance", "cumulative_amplitude",
    "peak_velocity", "median_velocity", "min_velocity",
    "is_outlier", "outlier_reasons",
    "num_fixs_to_strip",
]


def process_trial_events(
        all_eye_movement_features: pd.DataFrame, end_time: float,
) -> pd.DataFrame:
    """
    Tabulate every eye-movement event detected during the trial.

    :param all_eye_movement_features: DataFrame containing the eye movement features for the trial.
    :param end_time: float; the end time of the trial in ms (relative to trial onset).

    :return: one row per detected event, indexed by (eye, event), with the columns listed in `_COLUMN_ORDER`.
    """
    assert end_time > 0, f"Trial end time must be a positive number, got {end_time}."
    features = _extract_event_features(all_eye_movement_features, end_time)
    fixs_to_strip = _num_fixations_to_strip(features)
    events = pd.concat([features, fixs_to_strip], axis=1)
    return events[[col for col in _COLUMN_ORDER if col in events.columns]]


def _extract_event_features(trial_eye_movements: pd.DataFrame, trial_end_time: float) -> pd.DataFrame:
    """
    Reshape `peyes.summarize_events` output into the persisted event table.

    Three things happen here beyond a rename:

    - **The 2-tuple columns are split.** `center_pixel` and `pixel_std` are object-dtype tuples at ~100 B/cell
      against 16 B for two float64 columns, and neither is usable in a vectorised comparison as a tuple.
    - **`x`/`y` are NaN for non-fixations.** They mean "the position the eye was holding". A saccade's
      `center_pixel` is the midpoint of a trajectory the eye crossed at speed and never held; a blink's is the mean
      of missing or interpolated samples. Writing either into `x`/`y` would fabricate a gaze position that
      downstream code has every reason to treat as one. `start_*`/`end_*` carry the saccade geometry instead.
    - **`label` is dropped.** It is the integer form of `event_type`, which is kept.

    The raw spread features (`pixel_std` -> `std_x`/`std_y`, `dispersion`, `ellipse_area`) are kept for every event
    rather than nulled: they are measurements, not a fabricated location. They are only *interpretable* for
    fixations - for a saccade they describe the movement, not stability.
    """
    features = trial_eye_movements.copy()
    to_trial_end = (trial_end_time - features[cnst.END_TIME_STR]).rename("to_trial_end")
    is_fixation = features[cnst.LABEL_STR] == _FIXATION_LABEL
    centers = _split_tuple_column(features, "center_pixel", cnst.X, cnst.Y)
    centers = centers.where(is_fixation, other=np.nan)
    stds = _split_tuple_column(features, "pixel_std", "std_x", "std_y")
    features = pd.concat([features, to_trial_end, centers, stds], axis=1)
    features = (
        features
        .drop(columns=[cnst.LABEL_STR, "center_pixel", "pixel_std"], inplace=False, errors="ignore")
        .reset_index(drop=False, inplace=False)
    )
    features[_EVENT_TYPE_STR] = features[_EVENT_TYPE_STR].astype("category")
    return features


def _split_tuple_column(df: pd.DataFrame, source: str, x_name: str, y_name: str) -> pd.DataFrame:
    """Split an object column of (x, y) 2-tuples into two float columns."""
    if source not in df.columns:
        return pd.DataFrame(index=df.index, columns=[x_name, y_name], dtype=float)
    return pd.DataFrame(
        df[source].to_list(), index=df.index, columns=[x_name, y_name], dtype=float,
    )


def _num_fixations_to_strip(features: pd.DataFrame) -> pd.Series:
    """
    For each fixation, count how many fixations until one lands in the bottom strip of the SearchArray, or `inf` if
    none do. Non-fixation events get NaN.

    The count is over **fixations only** and computed **per eye**. Both matter, and both are positional traps: the
    frame now holds saccades and blinks interleaved with fixations, so scanning it as-is would count *events* to the
    strip rather than fixations; and it holds both eyes concatenated (all of the left eye's, then all of the right
    eye's - see `Trial.get_raw_eye_movements`), so a single scan would run off the end of one eye's events and into
    the other's, which restarts at the beginning of the trial.
    """
    if cnst.EYE_STR not in features.columns:
        raise KeyError(f"`features` must contain an '{cnst.EYE_STR}' column to count per eye.")
    fixations = features.loc[features[_EVENT_TYPE_STR] == cnst.FIXATION_STR.upper()]
    if fixations.empty:
        return pd.Series(np.nan, index=features.index, name="num_fixs_to_strip", dtype=float)
    is_in_strip = pd.Series(
        [SearchArray.is_in_bottom_strip((row.x, row.y)) for row in fixations[[cnst.X, cnst.Y]].itertuples()],
        index=fixations.index, name="is_in_strip", dtype=bool,
    )
    fixs_to_strip = (
        is_in_strip
        .groupby(fixations[cnst.EYE_STR], sort=False, observed=True)
        .transform(_num_to_next_true)
    )
    return fixs_to_strip.reindex(features.index).rename("num_fixs_to_strip")


def _num_to_next_true(bools: Sequence[bool]) -> pd.Series:
    """
    Distance from each element to the next True, or `inf` where no later True exists.

    Counts positionally, so callers must pass one contiguous, time-ordered sequence, but *returns* on the caller's
    index. The distinction only started to matter once the fixations became a subset of a larger table: the
    fixation rows no longer have a contiguous index, and `groupby.transform` aligns the result by label.
    """
    values = np.asarray(bools, dtype=bool)
    index = bools.index if isinstance(bools, pd.Series) else pd.RangeIndex(len(values))
    positions = np.arange(len(values))
    true_positions = np.flatnonzero(values)
    next_true_idx = np.searchsorted(true_positions, positions, side='left')
    dist_to_true = np.full(len(values), np.inf)
    valid = next_true_idx < len(true_positions)
    dist_to_true[valid] = true_positions[next_true_idx[valid]] - positions[valid]
    assert (dist_to_true >= 0).all(), "Distances should be non-negative."
    return pd.Series(dist_to_true, index=index, dtype=float)
