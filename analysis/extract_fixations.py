import numpy as np
import pandas as pd
import peyes

import config as cnfg
from data_models.Trial import Trial
from data_models.LWSEnums import DominantEyeEnum

_FIXATION_LABEL = peyes.parse_label("fixation")
_REDUNDANT_FEATURES = ['label', 'distance', 'velocity', 'amplitude', 'azimuth', 'dispersion', 'area', 'is_outlier']


def extract_fixations(trial: Trial) -> pd.DataFrame:
    """
    Processes fixations from a trial and returns a DataFrame indexed by (eye, fixation ID), containing the following
    information about each fixation:
    - start-time, end-time and duration: float (in ms)
    - center-pixel: tuple (x, y) - the mean pixel coordinates of the fixation
    - pixel_std: tuple (x, y) - the standard deviation of the pixel coordinates of the fixation
    - outlier_reasons: List[str] - reasons for the fixation being an outlier (or [] if not and outlier)
    - is_in_strip: bool - whether the fixation is in the bottom strip of the screen
    - all_marked: List[str] - all targets that were identified previously or during the current fixation
    - curr_marked: str - the target that was identified during the current fixation (or None)
    - target_0, target_1, ...: float - pixel-distances to each target in the trial
    """
    fixs_df = _fixations_to_frame(trial)
    is_in_strip = _is_in_strip(trial, fixs_df)
    marked_targets = _marked_targets(trial, fixs_df)
    pixel_distances = _calculate_distances(trial, fixs_df)
    res = pd.concat([fixs_df, is_in_strip, marked_targets, pixel_distances], axis=1)
    return res


def _fixations_to_frame(trial: Trial) -> pd.DataFrame:
    """ Extracts fixations from a trial and returns them as a DataFrame. """
    left_em = trial.get_eye_movements(eye=DominantEyeEnum.Left)
    left_fixs = list(filter(lambda e: e.label == _FIXATION_LABEL, left_em))
    left_fixs_df = peyes.summarize_events(left_fixs)
    right_em = trial.get_eye_movements(eye=DominantEyeEnum.Right)
    right_fixs = list(filter(lambda e: e.label == _FIXATION_LABEL, right_em))
    right_fixs_df = peyes.summarize_events(right_fixs)
    fixs_df = pd.concat(
        [left_fixs_df, right_fixs_df], keys=[cnfg.LEFT_STR, cnfg.RIGHT_STR],
        names=["eye", _FIXATION_LABEL.name.lower()], axis=0
    )
    # drop redundant columns
    to_drop = [col for feat in _REDUNDANT_FEATURES for col in fixs_df.columns if feat in col]
    fixs_df.drop(columns=to_drop, inplace=True, errors='ignore')
    return fixs_df


def _is_in_strip(trial: Trial, fixs_df: pd.DataFrame) -> pd.Series:
    """ Checks if fixations are in the bottom strip of the trial. """
    return fixs_df['center_pixel'].map(lambda p: trial.is_in_bottom_strip(p)).rename("is_in_strip")


def _marked_targets(trial: Trial, fixs_df: pd.DataFrame) -> pd.DataFrame:
    """ Identifies targets that were already identified or identified during the current fixation. """
    target_identification_data = trial.extract_target_identification()      # targets' identification time
    is_end_after = pd.DataFrame(
        fixs_df[cnfg.END_TIME_STR].values >= target_identification_data[cnfg.TIME_STR].values[:, np.newaxis],
        columns=fixs_df.index, index=target_identification_data.index
    ).T
    curr_and_prior_marked = is_end_after.apply(lambda row: set(row.index[row]), axis=1).rename("all_marked")
    curr_mark = curr_and_prior_marked.diff().map(
        lambda s: list(s)[0] if isinstance(s, set) and len(s) else None
    ).rename("curr_marked")
    marked = pd.concat([curr_and_prior_marked, curr_mark], axis=1)
    return marked

def _calculate_distances(trial: Trial, fixs_df: pd.DataFrame) -> pd.DataFrame:
    """ Calculates distances from fixations to targets. """
    center_pixels = np.vstack(fixs_df['center_pixel'].values)
    dists = trial.calculate_target_distances(center_pixels[:, 0], center_pixels[:, 1])
    dists.index = fixs_df.index
    return dists

