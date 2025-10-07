from typing import Sequence

import numpy as np
import pandas as pd
import peyes

import constants as cnst
from data_models.SearchArray import SearchArray


_FIXATION_LABEL = peyes.parse_label(cnst.FIXATION_STR)
_REDUNDANT_FIXATION_FEATURES = [
    'label', "center_pixel", "pixel_std", 'azimuth', 'is_outlier',
    "distance", "cumulative_distance", "amplitude", "cumulative_amplitude",
    'velocity', "peak_velocity", "median_velocity", "min_velocity",
    'dispersion', "area", "median_dispersion", "ellipse_area",
]


def process_trial_fixations(
        all_eye_movement_features: pd.DataFrame, targets: pd.DataFrame, end_time: float, px2deg: float,
) -> pd.DataFrame:
    """
    Processes the fixations of during the trial and returns them as a DataFrame.

    :param all_eye_movement_features: DataFrame containing the eye movement features for the trial.
    :param targets: DataFrame containing the target information for the trial.
    :param end_time: float; the end time of the trial in ms (relative to trial onset).
    :param px2deg: float; the conversion factor from pixels to degrees of visual angle (DVA).

    :return: a DataFrame containing the fixations during the trial, with the following columns:
    - eye: str; the eye that the fixation belongs to (left or right)
    - event: int; the number of the fixation among all events from the given eye during the trial
    - start_time: float; time of the fixation start in ms (relative to trial onset)
    - end_time: float; time of the fixation end in ms (relative to trial onset)
    - duration: float; duration of the fixation in ms
    - to_trial_end: float; time from the end of the fixation to the end of the trial in ms
    - x: float; x coordinates of the fixation center in pixels
    - y: float; y coordinates of the fixation center in pixels
    - outlier_reasons: List[str]; reasons for the fixation to be an outlier
    - target: str; the name of the closest target to the fixation center at the time of the fixation
    - target{i}_distance_px: float; the distance between the fixation center and target{i} in pixels (target0, target1, etc.)
    - target{i}_distance_dva: float; the distance between the fixation center and target{i} in DVA (target0, target1, etc.)
    - num_fixs_to_strip: int; number of fixations from the current fixation until a visit in the bottom strip of the
    SearchArray. Value is 0 if he current fixation is in the bottom strip, and np.inf if there are no future fixations
    in the strip during the trial.
    """
    assert end_time > 0, f"Trial end time must be a positive number, got {end_time}."
    assert px2deg > 0, f"Trial's `px2deg` conversion factor must be a positive number, got {px2deg}."
    fix_features = _extract_fixation_features(all_eye_movement_features, end_time)
    dists = _calc_target_distances(fix_features, targets, px2deg)
    closest_target = _find_closest_target(dists)
    fixs_to_strip = _num_fixations_to_strip(fix_features)
    fixations = pd.concat([fix_features, dists, closest_target, fixs_to_strip], axis=1)
    return fixations


def _extract_fixation_features(trial_eye_movements: pd.DataFrame, trial_end_time: float) -> pd.DataFrame:
    """
    Extract the trial's fixation features to a DataFrame containing the following columns:
    - eye: left/right
    - event: int; number of the fixation among all events in the trial from the given eye
    - start_time: float; time of the fixation start
    - end_time: float; time of the fixation end
    - duration: float; duration of the fixation in seconds
    - x: float; x coordinate of the fixation center in pixels
    - y: float; y coordinate of the fixation center in pixels
    - to_trial_end: float; time from the end of the fixation to the end of the trial in seconds
    - outlier_reasons: List[str]; reasons for the fixation to be an outlier
    """
    features = trial_eye_movements[trial_eye_movements[cnst.LABEL_STR] == _FIXATION_LABEL]
    to_trial_end = (trial_end_time - features[cnst.END_TIME_STR]).rename("to_trial_end")
    centers = pd.DataFrame(features["center_pixel"].to_list(), index=features.index, columns=[cnst.X, cnst.Y])
    features = pd.concat([features, to_trial_end, centers], axis=1)
    features = (
        features
        .drop(columns=_REDUNDANT_FIXATION_FEATURES, inplace=False, errors='ignore')
        .reset_index(drop=False, inplace=False)
    )
    return features


def _calc_target_distances(
        fix_features: pd.DataFrame, target_info: pd.DataFrame, px2deg: float,
) -> pd.DataFrame:
    """ Calculate the pixel-and DVA-distance between each fixation and each target in the trial. """
    fixs_xy = fix_features[[cnst.X, cnst.Y]].to_numpy()     # shape: (n_fixations, 2)
    tgts_xy = target_info[                                  # shape: (n_targets, 2)
        [f"{cnst.TARGET_STR}_{cnst.X}", f"{cnst.TARGET_STR}_{cnst.Y}"]
    ].to_numpy()
    dxy = fixs_xy[:, None, :] - tgts_xy[None, :, :]         # shape (num_fixations, num_targets, 2)
    distances = np.linalg.norm(dxy, axis=2)                 # shape (num_fixations, num_targets)
    distances_px = pd.DataFrame(distances, columns=target_info.index, index=fix_features.index)
    distances_dva = distances_px * px2deg                   # convert to DVA
    distances = pd.concat([
        distances_px.rename(columns=lambda tgt: f"{tgt}_{cnst.DISTANCE_STR}_px"),
        distances_dva.rename(columns=lambda tgt: f"{tgt}_{cnst.DISTANCE_STR}_dva")
    ], axis=1)
    return distances


def _find_closest_target(distances: pd.DataFrame) -> pd.Series:
    """ Find the closest target for each fixation in the trial """
    suffix = f"_{cnst.DISTANCE_STR}_dva"
    dist_cols = [col for col in distances.columns if col.endswith(suffix)]
    if len(dist_cols) == 0:
        suffix = f"_{cnst.DISTANCE_STR}_px"
        dist_cols = [col for col in distances.columns if col.endswith(suffix)]
    if len(dist_cols) == 0:
        raise RuntimeError(f"No distance columns found in the distances DataFrame: {distances.columns.tolist()}")
    closest_target = (
        distances
        .loc[:, dist_cols]
        .idxmin(axis=1)
        .map(lambda colname: colname.replace(suffix, ""))  # remove the distance suffix
        .rename(cnst.TARGET_STR)
    )
    return closest_target


def _num_fixations_to_strip(fix_features: pd.DataFrame) -> pd.Series:
    """ For each fixation, check how many fixations from it until a visit in the bottom strip of the SearchArray. """
    xy = fix_features[[cnst.X, cnst.Y]]      # shape (num_fixs, 2)
    is_in_strip = pd.Series(
        map(lambda tup: SearchArray.is_in_bottom_strip((tup.x, tup.y)), xy.itertuples()),
        name="is_in_strip", dtype=bool,
    )

    def num_to_true(bools: Sequence[bool]) -> pd.Series:
        """
        Converts a boolean Series or array to a Series of floats indicating the distance to the next True value, or inf
        if no future True exists.
        """
        bools = pd.Series(np.asarray(bools), dtype=bool)
        indices = np.arange(len(bools))
        true_indices = np.flatnonzero(bools.to_numpy())
        next_true_idx = np.searchsorted(true_indices, indices, side='left')
        dist_to_true = np.full(len(bools), np.inf)
        valid = next_true_idx < len(true_indices)
        dist_to_true[valid] = true_indices[next_true_idx[valid]] - indices[valid]
        assert (dist_to_true >= 0).all(), "Distances should be non-negative."
        return pd.Series(dist_to_true, index=bools.index, dtype=float)


    fixs_to_strip = num_to_true(is_in_strip).rename("num_fixs_to_strip")
    return fixs_to_strip


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
