from __future__ import annotations
import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import peyes

import config as cnfg
import helpers as hlp
from data_models.SearchArray import SearchArray
from data_models.LWSEnums import SearchArrayTypeEnum, SearchActionTypesEnum, DominantEyeEnum
from parse.eye_movements import detect_eye_movements

_FIXATION_LABEL = peyes.parse_label(cnfg.FIXATION_STR)
_REDUNDANT_FIXATION_FEATURES = [
    'label', 'distance', 'velocity', 'amplitude', 'azimuth', 'dispersion', 'area', 'is_outlier'
]
_MAX_GAZE_TO_TRIGGER_TIME_DIFF = 10  # in ms    # Maximum allowed time difference between gaze and trigger events for them to be considered as part of the same event.


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
    def trial_type(self) -> SearchArrayTypeEnum:
        return self._search_array.array_type

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

    @staticmethod
    def is_in_bottom_strip(p: Tuple[float, float]) -> bool:
        """ Check if a point is within the bottom strip rectangle, containing target exemplars. """
        return SearchArray.is_in_bottom_strip(p)

    def get_search_array(self) -> SearchArray:
        return self._search_array

    def get_triggers(self) -> pd.DataFrame:
        return self._triggers

    def get_gaze(self) -> pd.DataFrame:
        return self._gaze

    def get_actions(self) -> pd.DataFrame:
        """ Returns the times and actions performed by the subject during the trial. """
        return self._triggers.loc[self._triggers[cnfg.ACTION_STR].notnull(), [cnfg.TIME_STR, cnfg.ACTION_STR]]

    def get_metadata(self) -> pd.Series:
        return pd.Series({
            "trial_num": self.trial_num,
            "block_num": self.block_num,
            "trial_type": self.trial_type,
            "duration": self.end_time - self.start_time,
            "num_targets": len(self._search_array.targets),
        })

    def get_eye_movements(self, eye: DominantEyeEnum) -> pd.Series:
        if eye == DominantEyeEnum.Left:
            return self._left_events
        if eye == DominantEyeEnum.Right:
            return self._right_events
        raise ValueError(f"Invalid eye: {eye}. Must be either 'left' or 'right'.")

    def get_targets(self) -> pd.DataFrame:
        target_images = self._search_array.targets
        target_df = pd.DataFrame(target_images, index=[f"{cnfg.TARGET_STR}_{i}" for i in range(len(target_images))])
        target_df[cnfg.CATEGORY_STR] = [img.category for img in target_images]
        return target_df

    def calculate_target_distances(self, x: np.ndarray, y: np.ndarray,) -> pd.DataFrame:
        """
        Calculate the pixel-distance from each X-Y coordinate to each target in the search array.
        :param x: 1D array of X coordinates with shape (N,) or (N, 1) or (1, N)
        :param y: 1D array of Y coordinates with shape (N,) or (N, 1) or (1, N)
        :return: a (num_coords, num_targets) DataFrame with the distances from each coordinate to each target.
        """
        x = hlp.flatten_or_raise(x)
        y = hlp.flatten_or_raise(y)
        if x.shape != y.shape:
            raise ValueError(f"Input arrays must have the same shape. Got {x.shape} and {y.shape}.")
        coords = np.column_stack((x, y))                    # shape (n_coords, 2)
        targets = self.get_targets()
        target_coords = targets[[cnfg.X, cnfg.Y]].values    # shape (n_targets, 2)
        dists = np.empty((len(coords), len(targets)), dtype=float)
        for i, (cx, cy) in enumerate(coords):
            for j, (tx, ty) in enumerate(target_coords):
                dists[i, j] = hlp.distance((cx, cy), (tx, ty), 'px',)
        dists = pd.DataFrame(dists, columns=targets.index)
        return dists

    def extract_target_identification(self) -> pd.DataFrame:
        """
        Extracts the time, gaze location and pixel-distance from the target during target identifications.
        If a taget was never identified, the time and distance are set to np.inf and gaze data is set to NaN.
        If a target was identified multiple times, we record all identifications and raise a warning.

        :return: pd.DataFrame indexed by targets (`target_0`, `target_1`, etc.) with the following columns:
            - `time`: time of identification
            - `distance`: pixel-distance from the target at the time of identification
            - `left_x`, `left_y`: gaze coordinates in the left eye at the time of identification
            - `right_x`, `right_y`: gaze coordinates in the right eye at the time of identification
            - `left_pupil`, `right_pupil`: pupil size in the left/right eye at the time of identification
            - `'left_label'`, `'right_label'`: eye movement labels in the left/right eye at the time of identification
            - `target_x`, `target_y`: target coordinates
            - `target_category`: target category
        """
        # extract gaze on target identification
        identification_triggers = self._triggers[
            self._triggers[cnfg.ACTION_STR] == SearchActionTypesEnum.MARK_AND_CONFIRM
        ].reset_index(drop=True)
        gaze_when_ident = self._gaze.loc[hlp.closest_indices(
            self._gaze['time'], identification_triggers['time'], threshold=_MAX_GAZE_TO_TRIGGER_TIME_DIFF
        )].reset_index(drop=True)

        # calculate minimal distance from targets
        dists = gaze_when_ident[[col for col in gaze_when_ident.columns if col.startswith(cnfg.TARGET_STR)]].copy()
        closest_target = dists.idxmin(axis=1)
        dists = pd.Series(
            dists.to_numpy()[dists.index, dists.columns.get_indexer(closest_target)],
            name=f"{cnfg.DISTANCE_STR}_px",
        )

        # concatenate results
        res = pd.concat([
            identification_triggers[cnfg.TIME_STR].astype(float),
            dists,
            gaze_when_ident[[col for col in gaze_when_ident.columns if col.startswith(cnfg.LEFT_STR)]],
            gaze_when_ident[[col for col in gaze_when_ident.columns if col.startswith(cnfg.RIGHT_STR)]],
        ], axis=1)
        res.index = closest_target.values
        target_data = self.get_targets()[[cnfg.X, cnfg.Y, cnfg.CATEGORY_STR]].rename(
            columns=lambda col: f"{cnfg.TARGET_STR}_{col}", inplace=False
        )
        res = pd.concat([res, target_data], axis=1)

        # replace unidentified targets' `time` and `distance` with np.inf
        non_nan_cols = [cnfg.TIME_STR] + [col for col in res if col.startswith(cnfg.DISTANCE_STR)]
        res.loc[res[cnfg.TIME_STR].isna(), non_nan_cols] = np.inf

        # warn if a target was identified multiple times
        target_counts = res.index.value_counts()
        multi_detected = target_counts[target_counts > 1]
        if not multi_detected.empty:
            # TODO: consider resolving this in code rather than manually?
            warnings.warn(
                f"Multiple identifications for targets {multi_detected.index.tolist()} in trial {self.trial_num}.",
                RuntimeWarning,
            )
        return res

    def process_fixations(self) -> pd.DataFrame:
        """
        Processes the trial's fixations and returns a DataFrame indexed by (eye, fixation ID), containing the following
        information about each fixation:
        - start-time, end-time and duration: float (in ms)
        - center-pixel: tuple (x, y) - the mean pixel coordinates of the fixation
        - pixel_std: tuple (x, y) - the standard deviation of the pixel coordinates of the fixation
        - outlier_reasons: List[str] - reasons for the fixation being an outlier (or [] if not and outlier)
        - target_0, target_1, ...: float - pixel-distances to each target in the trial
        - all_marked: List[str] - all targets that were identified previously or during the current fixation
        - curr_marked: str - the target that was identified during the current fixation (or None)
        - in_strip: bool - whether the fixation is in the bottom strip of the trial
        - time_to_trial_end: float - time from fixation's end to the end of the trial (in ms)
        """
        left_em = self.get_eye_movements(eye=DominantEyeEnum.Left)
        left_fixs = list(filter(lambda e: e.label == _FIXATION_LABEL, left_em))
        left_fixs_df = peyes.summarize_events(left_fixs)
        right_em = self.get_eye_movements(eye=DominantEyeEnum.Right)
        right_fixs = list(filter(lambda e: e.label == _FIXATION_LABEL, right_em))
        right_fixs_df = peyes.summarize_events(right_fixs)
        fixs_df = pd.concat(
            [left_fixs_df, right_fixs_df], keys=[cnfg.LEFT_STR, cnfg.RIGHT_STR],
            names=["eye", cnfg.FIXATION_STR], axis=0
        )
        fixs_df.drop(  # drop redundant columns
            columns=[col for feat in _REDUNDANT_FIXATION_FEATURES for col in fixs_df.columns if feat in col],
            inplace=True, errors='ignore'
        )

        # calculate distances from fixations to targets
        center_pixels = np.vstack(fixs_df['center_pixel'].values)
        dists = self.calculate_target_distances(center_pixels[:, 0], center_pixels[:, 1])
        dists.index = fixs_df.index

        # identifies targets that were already identified or identified during the current fixation
        target_identification_data = self.extract_target_identification()  # targets' identification time
        is_end_after = pd.DataFrame(
            fixs_df[cnfg.END_TIME_STR].values >= target_identification_data[cnfg.TIME_STR].values[:, np.newaxis],
            columns=fixs_df.index, index=target_identification_data.index
        ).T
        curr_and_prior_marked = is_end_after.apply(lambda row: set(row.index[row]), axis=1).rename("all_marked")
        curr_mark = curr_and_prior_marked.diff().map(
            lambda s: list(s)[0] if isinstance(s, set) and len(s) else None
        ).rename("curr_marked")
        marked = pd.concat([curr_and_prior_marked, curr_mark], axis=1)

        # checks if the fixation is in the bottom strip of the trial
        in_strip = fixs_df['center_pixel'].map(lambda p: self.is_in_bottom_strip(p)).rename("in_strip")

        # calculate the time to trial's end
        time_to_trial_end = fixs_df[cnfg.END_TIME_STR].map(lambda t: self.end_time - t).rename("time_to_trial_end")

        # concatenate all data into a single DataFrame
        fixs_df = pd.concat([fixs_df, dists, marked, in_strip, time_to_trial_end], axis=1)
        return fixs_df

    def _create_search_array(self) -> SearchArray:
        search_array_type = SearchArrayTypeEnum[_extract_singleton_column(self._gaze, cnfg.CONDITION_STR).upper()]
        search_array_num = int(_extract_singleton_column(self._gaze, "image_num"))
        search_array = SearchArray.from_mat(os.path.join(
            cnfg.SEARCH_ARRAY_PATH,
            f"generated_stim{cnfg.STIMULI_VERSION}",
            search_array_type.name.lower(),
            f"image_{search_array_num}.mat"
        ))
        return search_array

    def _detect_eye_movements(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        left_labels, left_events = detect_eye_movements(
            self._gaze,
            DominantEyeEnum.Left,
            self._subject.screen_distance_cm,
            cnfg.DETECTOR,
            cnfg.TOBII_PIXEL_SIZE_MM / 10,
            only_labels=False
        )
        right_labels, right_events = detect_eye_movements(
            self._gaze,
            DominantEyeEnum.Right,
            self._subject.screen_distance_cm,
            cnfg.DETECTOR,
            cnfg.TOBII_PIXEL_SIZE_MM / 10,
            only_labels=False
        )
        labels = pd.concat([left_labels, right_labels], axis=1)
        return labels, left_events, right_events

    def _calculate_gaze_target_distances(self,) -> pd.DataFrame:
        left_dists = self.calculate_target_distances(
            self._gaze[cnfg.LEFT_X_STR].values, self._gaze[cnfg.LEFT_Y_STR].values
        )
        right_dists = self.calculate_target_distances(
            self._gaze[cnfg.RIGHT_X_STR].values, self._gaze[cnfg.RIGHT_Y_STR].values
        )
        main = left_dists if self._subject.eye == DominantEyeEnum.Left else right_dists
        second = right_dists if self._subject.eye == DominantEyeEnum.Left else left_dists
        dists = main.fillna(second)
        dists.index = self._gaze.index
        return dists

    def __eq__(self, other) -> bool:
        if not isinstance(other, Trial):
            return False
        if self.trial_num != other.trial_num:
            return False
        if self.block_num != other.block_num:
            return False
        if self.trial_type != other.trial_type:
            return False
        if self.get_search_array() != other.get_search_array():
            return False
        return True

    def __repr__(self) -> str:
        return f"Trial {self.trial_num} ({self.get_search_array().array_type.name})"
