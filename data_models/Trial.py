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
from data_models.LWSEnums import SearchArrayTypeEnum, SubjectActionTypesEnum, DominantEyeEnum
from parse.eye_movements import detect_eye_movements

_FIXATION_LABEL = peyes.parse_label(cnfg.FIXATION_STR)
_REDUNDANT_FIXATION_FEATURES = [
    'label', 'distance', 'velocity', 'amplitude', 'azimuth', 'dispersion', 'area', 'is_outlier'
]
_MAX_GAZE_TO_TRIGGER_TIME_DIFF = 10  # in ms    # Maximum allowed time difference between gaze and trigger events for them to be considered as part of the same event.
_BAD_SUBJECT_ACTIONS = [
    SubjectActionTypesEnum.MARK_ONLY, SubjectActionTypesEnum.ATTEMPTED_MARK, SubjectActionTypesEnum.MARK_AND_REJECT
]


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

    @property
    def is_bad(self) -> bool:
        """ Returns True if the trial is considered 'bad', i.e. contains bad subject actions. """
        # TODO: add logic to exclude more trials
        return bool(np.isin(self.get_actions()[cnfg.ACTION_STR], _BAD_SUBJECT_ACTIONS).any())

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
            "is_bad": self.is_bad,
        })

    def get_eye_movements(self, eye: DominantEyeEnum) -> pd.Series:
        if eye == DominantEyeEnum.LEFT:
            return self._left_events
        if eye == DominantEyeEnum.RIGHT:
            return self._right_events
        raise ValueError(f"Invalid eye: {eye}. Must be either 'left' or 'right'.")

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
        target_coords = np.array([(img.x, img.y) for img in self._search_array.targets])  # shape (n_targets, 2)
        dists = np.empty((len(coords), len(target_coords)), dtype=float)
        for i, (cx, cy) in enumerate(coords):
            for j, (tx, ty) in enumerate(target_coords):
                dists[i, j] = hlp.distance((cx, cy), (tx, ty), 'px',)
        dists = pd.DataFrame(dists, columns=[f"{cnfg.TARGET_STR}_{i}" for i in range(target_coords.shape[0])])
        return dists

    def get_target_identification_data(self) -> pd.DataFrame:
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
            - `target_angle`: target rotation angle
            - `target_sub_path`: path to the target image
            - `target_category`: target category
        """
        # extract gaze on target identification
        identification_triggers = self._triggers[
            self._triggers[cnfg.ACTION_STR] == SubjectActionTypesEnum.MARK_AND_CONFIRM
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

        # extract the target data
        target_images = self._search_array.targets
        target_df = pd.DataFrame(target_images, index=[f"{cnfg.TARGET_STR}_{i}" for i in range(len(target_images))])
        target_df[cnfg.CATEGORY_STR] = [img.category for img in target_images]
        target_df = target_df.rename(columns=lambda col: f"{cnfg.TARGET_STR}_{col}", inplace=False)

        # concatenate results
        res = pd.concat([
            identification_triggers[cnfg.TIME_STR].astype(float),
            dists,
            gaze_when_ident[[col for col in gaze_when_ident.columns if col.startswith(cnfg.LEFT_STR)]],
            gaze_when_ident[[col for col in gaze_when_ident.columns if col.startswith(cnfg.RIGHT_STR)]],
        ], axis=1)
        res.index = closest_target.values
        res = pd.concat([res, target_df], axis=1)
        res.index.names = [cnfg.TARGET_STR]

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
        - from_trial_start: float - time from trial's start to the start of the fixation (in ms)
        - to_trial_end: float - time from fixation's end to the end of the trial (in ms)
        """
        left_em = self.get_eye_movements(eye=DominantEyeEnum.LEFT)
        left_fixs = list(filter(lambda e: e.label == _FIXATION_LABEL, left_em))
        left_fixs_df = peyes.summarize_events(left_fixs)
        right_em = self.get_eye_movements(eye=DominantEyeEnum.RIGHT)
        right_fixs = list(filter(lambda e: e.label == _FIXATION_LABEL, right_em))
        right_fixs_df = peyes.summarize_events(right_fixs)
        fixs_df = pd.concat(
            [left_fixs_df, right_fixs_df],
            keys=[cnfg.LEFT_STR, cnfg.RIGHT_STR],
            names=[cnfg.EYE_STR, cnfg.FIXATION_STR],
            axis=0
        )
        fixs_df.drop(  # drop redundant columns
            columns=[col for feat in _REDUNDANT_FIXATION_FEATURES for col in fixs_df.columns if feat in col],
            inplace=True, errors='ignore'
        )

        # calculate distances from fixations to targets
        center_pixels = np.vstack(fixs_df['center_pixel'].values)
        dists = self.calculate_target_distances(center_pixels[:, 0], center_pixels[:, 1])
        dists.index = fixs_df.index

        # see if a target was marked during the current fixation
        target_identification_data = self.get_target_identification_data()  # targets' identification time
        is_start_before = pd.DataFrame(
            fixs_df[cnfg.START_TIME_STR].values <= target_identification_data[cnfg.TIME_STR].values[:, np.newaxis],
            columns=fixs_df.index, index=target_identification_data.index
        ).T
        is_end_after = pd.DataFrame(
            fixs_df[cnfg.END_TIME_STR].values >= target_identification_data[cnfg.TIME_STR].values[:, np.newaxis],
            columns=fixs_df.index, index=target_identification_data.index
        ).T
        is_marking = is_start_before & is_end_after  # fixation starts before and ends after the target identification time
        if not is_marking[is_marking.sum(axis=1) > 1].empty:
            # TODO: consider resolving this in code rather than manually?
            warnings.warn(
                f"Multiple targets marked during the same fixation in trial {self.trial_num}. "
                "This is not expected and may indicate an error in the data. "
                f"Erroneous fixations: {is_marking[is_marking.sum(axis=1) > 1].index.tolist()}",
                RuntimeWarning,
            )

        # set the "currently marked" column and the "currently or previously marked" column
        marked_targets = is_marking[is_marking.any(axis=1)].idxmax(axis=1)  # pd.Series of len num_marking_fixs
        curr_marked = pd.Series(None, index=fixs_df.index, name="curr_marked", dtype=str)
        curr_marked.loc[marked_targets.index] = marked_targets.values  # set the currently marked target for marking fixations
        curr_and_prior_marked = is_end_after.apply(lambda row: set(row.index[row]), axis=1).rename("all_marked")

        # checks if the fixation is in the bottom strip of the trial
        in_strip = fixs_df['center_pixel'].map(lambda p: self.is_in_bottom_strip(p)).rename("in_strip")

        # calculate the time from trial's start and to its end
        time_from_trial_start = fixs_df[cnfg.START_TIME_STR].map(lambda t: t - self.start_time).rename("from_trial_start")
        time_to_trial_end = fixs_df[cnfg.END_TIME_STR].map(lambda t: self.end_time - t).rename("to_trial_end")

        # concatenate all data into a single DataFrame
        fixs_df = pd.concat([
            fixs_df, dists, curr_marked, curr_and_prior_marked, in_strip, time_from_trial_start, time_to_trial_end
        ], axis=1)
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
            DominantEyeEnum.LEFT,
            self._subject.screen_distance_cm,
            cnfg.DETECTOR,
            cnfg.PIXEL_SIZE_MM / 10,
            only_labels=False
        )
        right_labels, right_events = detect_eye_movements(
            self._gaze,
            DominantEyeEnum.RIGHT,
            self._subject.screen_distance_cm,
            cnfg.DETECTOR,
            cnfg.PIXEL_SIZE_MM / 10,
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
        main = left_dists if self._subject.eye == DominantEyeEnum.LEFT else right_dists
        second = right_dists if self._subject.eye == DominantEyeEnum.LEFT else left_dists
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
