from __future__ import annotations
import os
import warnings
from typing import Tuple, Sequence

import numpy as np
import pandas as pd
import peyes

import config as cnfg
from data_models.SearchArray import SearchArray
from data_models.LWSEnums import SearchArrayCategoryEnum, SubjectActionCategoryEnum, DominantEyeEnum


def _extract_singleton_column(df: pd.DataFrame, col_name: str):
    values = df[col_name].dropna()
    assert values.nunique() == 1, f"Input data contains multiple values in column {col_name}"
    return values.iloc[0]


class Trial:
    """
    A class to represent a single LWS trial.
    Each trial consists of its SearchArray and behavioral data (pd.DataFrame).
    """

    def __init__(self, subject: "Subject", triggers: pd.DataFrame, gaze: pd.DataFrame,):
        # verify block number
        triggers_block_num = int(_extract_singleton_column(triggers, cnfg.BLOCK_STR))
        gaze_block_num = int(_extract_singleton_column(gaze, cnfg.BLOCK_STR))
        assert triggers_block_num == gaze_block_num, f"Triggers block num {triggers_block_num} does not match gaze block num {gaze_block_num}."

        # verify trial number
        triggers_trial_num = int(_extract_singleton_column(triggers, cnfg.TRIAL_STR))
        gaze_trial_num = int(_extract_singleton_column(gaze, cnfg.TRIAL_STR))
        assert triggers_trial_num == gaze_trial_num, f"Triggers trial num {self.trial_num} does not match gaze trial num {gaze_trial_num}."

        # store unprocessed data
        self._subject = subject
        self._triggers = triggers
        self._gaze = gaze

        # pre-process inputs
        self._search_array = self._create_search_array()
        dists = self._calculate_gaze_target_distances()
        labels, left_events, right_events = self._detect_eye_movements()
        self._gaze = pd.concat([self._gaze, labels, dists], axis=1)
        self._left_events = left_events
        self._right_events = right_events

    @property
    def block_num(self) -> int:
        return int(_extract_singleton_column(self._gaze, cnfg.BLOCK_STR))

    @property
    def trial_num(self) -> int:
        return int(_extract_singleton_column(self._gaze, cnfg.TRIAL_STR))

    @property
    def px2deg(self) -> float:
        """
        Returns the conversion factor from pixels to degrees of visual angle (DVA).
        To move from `d` pixels to DVA, use the formula: `d * self.px2deg`.
        """
        return self._subject.px2deg

    @property
    def start_time(self) -> float:
        gaze_min_time = self._gaze[cnfg.TIME_STR].min()
        triggers_min_time = self._triggers[cnfg.TIME_STR].min()
        return np.nanmin([gaze_min_time, triggers_min_time])

    @property
    def end_time(self) -> float:
        gaze_max_time = self._gaze[cnfg.TIME_STR].max()
        triggers_max_time = self._triggers[cnfg.TIME_STR].max()
        return np.nanmax([gaze_max_time, triggers_max_time])

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def trial_category(self) -> SearchArrayCategoryEnum:
        return self._search_array.array_category

    @property
    def num_targets(self):
        return len(self._search_array.targets)

    @property
    def num_distractors(self):
        return self._search_array.num_distractors

    @property
    def gaze_coverage(self) -> float:
        """ returns the percent of samples with valid gaze data (not NaN) for the dominant eye. """
        if self._subject.eye == DominantEyeEnum.LEFT:
            return self._calculate_gaze_coverage(DominantEyeEnum.LEFT)
        elif self._subject.eye == DominantEyeEnum.RIGHT:
            return self._calculate_gaze_coverage(DominantEyeEnum.RIGHT)
        else:
            raise NotImplementedError

    @property
    def num_actions(self) -> int:
        return len(self.get_actions())

    def get_gaze(self) -> pd.DataFrame:
        return self._gaze

    def get_actions(self) -> pd.DataFrame:
        """ Returns the times and actions performed by the subject during the trial. """
        triggers = self._triggers
        actions = (
            triggers
            .loc[triggers[cnfg.ACTION_STR].notnull()]
            .loc[triggers[cnfg.ACTION_STR] != SubjectActionCategoryEnum.NO_ACTION]
            .loc[:, [cnfg.TIME_STR, cnfg.ACTION_STR]]
        )
        to_trial_end = (self.end_time - actions[cnfg.TIME_STR]).rename("to_trial_end")
        actions = pd.concat([actions, to_trial_end], axis=1)
        return actions

    def get_targets(self) -> pd.DataFrame:
        """ Extracts the trial's target information: the targets' pixel coordinates, angle, category, and image path. """
        target_images = self._search_array.targets
        target_df = pd.DataFrame(target_images, index=[f"{cnfg.TARGET_STR}{i}" for i in range(len(target_images))])
        target_df[cnfg.CATEGORY_STR] = [img.category for img in target_images]
        target_df = target_df.rename(columns=lambda col: f"{cnfg.TARGET_STR}_{col}", inplace=False)
        return target_df

    def get_metadata(self, bad_actions: Sequence[SubjectActionCategoryEnum]) -> pd.Series:
        return pd.Series({
            "trial": self.trial_num,
            "block": self.block_num,
            "trial_category": self.trial_category.name,
            "duration": self.duration,
            "num_targets": self.num_targets,
            "num_distractors": self.num_distractors,
            "num_actions": self.num_actions,
            "bad_actions": bool(np.isin(self.get_actions()[cnfg.ACTION_STR], bad_actions).any()),
            "gaze_coverage": self.gaze_coverage,
        })

    def get_raw_eye_movements(self) -> pd.DataFrame:
        """ Returns a DataFrame summarizing the eye movements detected during the trial. """
        left = peyes.summarize_events(self._left_events)
        right = peyes.summarize_events(self._right_events)
        df = pd.concat(
            [left, right],
            keys=[cnfg.LEFT_STR, cnfg.RIGHT_STR],
            names=[cnfg.EYE_STR, cnfg.EVENT_STR], axis=0
        )
        return df

    def process_fixations(self) -> pd.DataFrame:
        from data_models.preprocess.fixations import process_trial_fixations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            features = self.get_raw_eye_movements()
        fixations = process_trial_fixations(features, self.get_targets(), self.end_time, self.px2deg)
        return fixations

    def _create_search_array(self) -> SearchArray:
        search_array_type = SearchArrayCategoryEnum[_extract_singleton_column(self._gaze, cnfg.CONDITION_STR).upper()]
        search_array_num = int(_extract_singleton_column(self._gaze, "image_num"))
        search_array = SearchArray.from_mat(os.path.join(
            cnfg.SEARCH_ARRAY_PATH,
            f"generated_stim{cnfg.STIMULI_VERSION}",
            search_array_type.name.lower(),
            f"image_{search_array_num}.mat"
        ))
        return search_array

    def _detect_eye_movements(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        from data_models.parse.eye_movements import detect_eye_movements
        left_labels, left_events = detect_eye_movements(
            self._gaze,
            DominantEyeEnum.LEFT,
            self._subject.screen_distance_cm,
            pixel_size_cm=cnfg.PIXEL_SIZE_MM / 10,
            only_labels=False
        )
        right_labels, right_events = detect_eye_movements(
            self._gaze,
            DominantEyeEnum.RIGHT,
            self._subject.screen_distance_cm,
            pixel_size_cm=cnfg.PIXEL_SIZE_MM / 10,
            only_labels=False
        )
        labels = pd.concat([left_labels, right_labels], axis=1)
        return labels, left_events, right_events

    def _calculate_gaze_target_distances(self,) -> pd.DataFrame:
        left_dists = self._calculate_target_distances(
            self._gaze[cnfg.LEFT_X_STR].values, self._gaze[cnfg.LEFT_Y_STR].values
        )
        right_dists = self._calculate_target_distances(
            self._gaze[cnfg.RIGHT_X_STR].values, self._gaze[cnfg.RIGHT_Y_STR].values
        )
        main = left_dists if self._subject.eye == DominantEyeEnum.LEFT else right_dists
        second = right_dists if self._subject.eye == DominantEyeEnum.LEFT else left_dists
        dists = main.fillna(second)
        dists.index = self._gaze.index
        return dists

    def _calculate_target_distances(self, x: np.ndarray, y: np.ndarray,) -> pd.DataFrame:
        """
        Calculate the pixel-distance from each X-Y coordinate to each target in the search array.
        :param x: 1D array of X coordinates with shape (N,) or (N, 1) or (1, N)
        :param y: 1D array of Y coordinates with shape (N,) or (N, 1) or (1, N)
        :return: a (num_coords, num_targets) DataFrame with the distances from each coordinate to each target.
        """
        if x.shape != y.shape:
            raise ValueError(f"Input arrays must have the same shape. Got {x.shape} and {y.shape}.")
        coords = np.column_stack((x, y))                                                            # shape (n_coords, 2)
        target_coords = np.array([(img.x, img.y) for img in self._search_array.targets])            # shape (n_targets, 2)
        dists = np.linalg.norm(coords[:, np.newaxis, :] - target_coords[np.newaxis, :, :], axis=2)  # shape (n_coords, n_targets)
        dists = pd.DataFrame(dists, columns=[f"{cnfg.TARGET_STR}{i}" for i in range(target_coords.shape[0])])
        return dists

    def _calculate_gaze_coverage(self, eye: DominantEyeEnum) -> float:
        """ Calculates the percent of samples with valid gaze data (not NaN) for the specified eye. """
        if len(self._gaze) <= 0:
            return np.nan
        x_str = cnfg.LEFT_X_STR if eye == DominantEyeEnum.LEFT else cnfg.RIGHT_X_STR
        y_str = cnfg.LEFT_Y_STR if eye == DominantEyeEnum.LEFT else cnfg.RIGHT_Y_STR
        valid_samples = self._gaze[x_str].notna() & self._gaze[y_str].notna()
        if len(self._gaze) > 0:
            return round(100 * valid_samples.sum() / len(self._gaze), 3)
        return np.nan

    def __eq__(self, other) -> bool:
        if not isinstance(other, Trial):
            return False
        if self.trial_num != other.trial_num:
            return False
        if self.block_num != other.block_num:
            return False
        if self.px2deg != other.px2deg:
            return False
        if self.start_time != other.start_time or self.end_time != other.end_time:
            return False
        if self.num_targets != other.num_targets:
            return False
        if self.num_distractors != other.num_distractors:
            return False
        if self.trial_category != other.trial_category:
            return False
        if self._search_array.version != other._search_array.version:
            return False
        if self._search_array.image_num != other._search_array.image_num:
            return False
        return True

    def __repr__(self) -> str:
        return f"Trial {self.trial_num} ({self.trial_category.name})"
