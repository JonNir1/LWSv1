"""Tabulate a trial's detected eye-movement events.

This module used to be `preprocess/fixations.py` and kept only the fixations, discarding saccades, blinks and 14
feature columns. It now keeps **every** detected event: the fixations are recoverable as
`events[event_type == "FIXATION"]`, and the saccades that were previously thrown away are what the planned
scanpath, amplitude and micro-saccade analyses need.

Two columns are populated for fixations only - see `_extract_event_features` for why - and the per-target distance
columns that used to live here are gone; see `_closest_target` and `CODE_REVIEW.md` on `fixations_to_targets()`.
"""

from typing import Sequence

import numpy as np
import pandas as pd
import peyes

import constants as cnst
from data_models.SearchArray import SearchArray


_FIXATION_LABEL = peyes.parse_label(cnst.FIXATION_STR)
_EVENT_TYPE_STR = f"{cnst.EVENT_STR}_type"
_CLOSEST_ICON_STR = f"closest_{cnst.ICON_STR}"
_CLOSEST_ICON_DISTANCE_STR = f"{_CLOSEST_ICON_STR}_{cnst.DISTANCE_DVA_STR}"

# columns whose values only make sense as a held gaze position, so they are NaN for non-fixations
_FIXATION_ONLY_COLUMNS = [cnst.X, cnst.Y, _CLOSEST_ICON_STR, _CLOSEST_ICON_DISTANCE_STR, "num_fixs_to_strip"]

_COLUMN_ORDER = [
    cnst.EYE_STR, cnst.EVENT_STR, _EVENT_TYPE_STR,
    cnst.START_TIME_STR, cnst.END_TIME_STR, "duration", "to_trial_end",
    cnst.X, cnst.Y, "start_x", "start_y", "end_x", "end_y",
    "std_x", "std_y", "dispersion", "ellipse_area",
    cnst.DISTANCE_STR, "amplitude", "azimuth", "cumulative_distance", "cumulative_amplitude",
    "peak_velocity", "median_velocity", "min_velocity",
    "is_outlier", "outlier_reasons",
    _CLOSEST_ICON_STR, _CLOSEST_ICON_DISTANCE_STR, "num_fixs_to_strip",
]


def process_trial_events(
        all_eye_movement_features: pd.DataFrame, targets: pd.DataFrame, end_time: float, px2deg: float,
) -> pd.DataFrame:
    """
    Tabulate every eye-movement event detected during the trial.

    :param all_eye_movement_features: DataFrame containing the eye movement features for the trial.
    :param targets: DataFrame containing the target information for the trial.
    :param end_time: float; the end time of the trial in ms (relative to trial onset).
    :param px2deg: float; the conversion factor from pixels to degrees of visual angle (DVA).

    :return: one row per detected event, indexed by (eye, event), with the columns listed in `_COLUMN_ORDER`:
    - eye: str; the eye the event was detected in (left or right)
    - event: int; the event's position among *all* events from that eye during the trial
    - event_type: str; FIXATION / SACCADE / BLINK (the only labels the Engbert detector emits)
    - start_time, end_time, duration: float; ms relative to trial onset
    - to_trial_end: float; time from the end of the event to the end of the trial, in ms
    - x, y: float; the held gaze position, **fixations only** (NaN otherwise)
    - start_x, start_y, end_x, end_y: float; the event's first and last gaze sample
    - std_x, std_y, dispersion, ellipse_area: float; spread of the event's samples (interpretable for fixations)
    - distance, amplitude, azimuth, cumulative_distance, cumulative_amplitude: float; trajectory geometry
    - peak_velocity, median_velocity, min_velocity: float; kinematics
    - is_outlier: bool; outlier_reasons: List[str]
    - closest_icon: str; the `icon{i}` identifier of the nearest **target**, **fixations only**
    - closest_icon_distance_dva: float; that target's distance in DVA, **fixations only**
    - num_fixs_to_strip: float; number of fixations until one lands in the bottom strip of the SearchArray
      (0 if this fixation is in the strip, `inf` if none ever is), **fixations only**
    """
    assert end_time > 0, f"Trial end time must be a positive number, got {end_time}."
    assert px2deg > 0, f"Trial's `px2deg` conversion factor must be a positive number, got {px2deg}."
    features = _extract_event_features(all_eye_movement_features, end_time)
    closest_target = _closest_target(features, targets, px2deg)
    fixs_to_strip = _num_fixations_to_strip(features)
    events = pd.concat([features, closest_target, fixs_to_strip], axis=1)
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


def _closest_target(features: pd.DataFrame, target_info: pd.DataFrame, px2deg: float) -> pd.DataFrame:
    """
    The nearest **target** to each fixation, by its stable `icon{i}` identifier, and its distance in DVA.

    This replaces the wide `icon{i}_distance_dva` / `_px` block that used to be persisted per event. Those columns
    are unique per trial - a trial's targets are not the next trial's - so concatenating subjects produced a
    97.3%-NaN table at ~1.4 kB/row. Per-target distances are cheap to recompute from `icons.pkl` and belong in a
    stage-2 `fixations_to_targets()` helper; see `CODE_REVIEW.md`.

    Despite the name, the candidates here are the trial's **targets**, not all 180 icons: only targets are available
    at this point in the pipeline. The identifier is an `icon{i}` because that is the icon's identity everywhere.
    """
    xy = features[[cnst.X, cnst.Y]].to_numpy(dtype=float)                # (n_events, 2)
    target_xy = target_info[                                             # (n_targets, 2)
        [f"{cnst.TARGET_STR}_{cnst.X}", f"{cnst.TARGET_STR}_{cnst.Y}"]
    ].to_numpy(dtype=float)
    if target_xy.size == 0:
        raise RuntimeError("trial has no targets, so no closest target can be assigned")
    distances_px = np.linalg.norm(xy[:, None, :] - target_xy[None, :, :], axis=2)    # (n_events, n_targets)
    has_position = np.isfinite(xy).all(axis=1)                           # False for every non-fixation
    nearest = np.full(len(features), -1, dtype=int)
    nearest[has_position] = distances_px[has_position].argmin(axis=1)
    target_ids = np.asarray(target_info.index, dtype=object)
    closest = pd.Series(None, index=features.index, name=_CLOSEST_ICON_STR, dtype=object)
    closest.loc[has_position] = target_ids[nearest[has_position]]
    closest_distance = pd.Series(
        np.where(has_position, distances_px[np.arange(len(features)), nearest] * px2deg, np.nan),
        index=features.index, name=_CLOSEST_ICON_DISTANCE_STR, dtype=float,
    )
    return pd.concat([closest, closest_distance], axis=1)


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

    Counts positionally, so callers must pass one contiguous, time-ordered sequence - but *returns* on the caller's
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


### Removing this to postpone the HIT/FA classification out of fixation extraction
# TODO: consider replacing with `number of fixation to/from identification`
# def _currently_identified_target(
#         fix_features: pd.DataFrame, behavior: pd.DataFrame, trial_num: int, on_target_threshold_dva: float
# ) -> pd.Series:
#     """ For each fixation, identify the target that was identified (hit) during that fixation, if any. """
#     assert np.isfinite(on_target_threshold_dva) and on_target_threshold_dva > 0, \
#         f"On-target threshold must be a finite positive number, got {on_target_threshold_dva}."
#
#     # identify which target (if any) was identified during each fixation
#     identified_targets = behavior[behavior[cnst.DISTANCE_DVA_STR] <= on_target_threshold_dva]
#     ident_times = identified_targets[cnst.TIME_STR].to_numpy()                              # shape: (num_hits,)
#     is_start_before = fix_features[cnst.START_TIME_STR].to_numpy() <= ident_times[:, None]  # shape (num_hits, num_fixations)
#     is_end_after = fix_features[cnst.END_TIME_STR].to_numpy() >= ident_times[:, None]       # shape (num_hits, num_fixations)
#     is_currently_identifying = is_start_before & is_end_after
#     is_currently_identifying = pd.DataFrame(                                                # shape (num_fixations, num_hits)
#         is_currently_identifying, index=identified_targets[cnst.TARGET_STR].to_list(), columns=fix_features.index,
#     ).T
#
#     # check for multiple targets identified during the same fixation - should not happen
#     simultaneous_identifications = is_currently_identifying[is_currently_identifying.sum(axis=1) > 1]
#     if not simultaneous_identifications.empty:
#         # TODO: consider resolving this in code rather than manually?
#         warnings.warn(
#             f"Multiple targets identified during the same fixation int trial {trial_num}. "
#             "This is not expected and may indicate an error in the data.",
#             RuntimeWarning,
#         )
#
#     # set the currently identified targets during identification fixations (we have at most `num_hits` such fixations)
#     identified_target = is_currently_identifying.loc[is_currently_identifying.any(axis=1)].idxmax(axis=1)   # len <= num_hits
#     curr_ident = pd.Series(None, index=fix_features.index, name="curr_identified", dtype=str)
#     curr_ident.loc[identified_target.index] = identified_target.values
#     return curr_ident
