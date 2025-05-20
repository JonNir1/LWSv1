from __future__ import annotations
import os
import warnings
from typing import Optional, Literal, Tuple

import numpy as np
import pandas as pd

import config as cnfg
import helpers as hlp
from data_models.SearchArray import SearchArray
from data_models.LWSEnums import SearchArrayTypeEnum, SearchActionTypesEnum, DominantEyeEnum
from parse.eye_movements import detect_eye_movements


def _extract_singleton_column(df: pd.DataFrame, col_name: str):
    values = df[col_name].dropna()
    assert values.nunique() == 1, f"Input data contains multiple values in column {col_name}"
    return values.iloc[0]


class Trial:
    """
    A class to represent a single LWS trial.
    Each trial consists of its SearchArray and behavioral data (pd.DataFrame).
    """

    def __init__(
            self,
            subject: "Subject",
            triggers: pd.DataFrame,
            gaze: pd.DataFrame,
            distance_unit: Literal['px', 'cm', 'deg'] = 'px'
    ):
        # verify block number
        triggers_block_num = int(_extract_singleton_column(triggers, cnfg.BLOCK_STR))
        gaze_block_num = int(_extract_singleton_column(gaze, cnfg.BLOCK_STR))
        assert triggers_block_num == gaze_block_num, f"Triggers block num {triggers_block_num} does not match gaze block num {gaze_block_num}."

        # verify trial number
        triggers_trial_num = int(_extract_singleton_column(triggers, cnfg.TRIAL_STR))
        gaze_trial_num = int(_extract_singleton_column(gaze, cnfg.TRIAL_STR))
        assert triggers_trial_num == gaze_trial_num, f"Triggers trial num {self.trial_num} does not match gaze trial num {gaze_trial_num}."

        # store unprocessed data
        self._distance_unit: Literal['px', 'cm', 'deg'] = distance_unit
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
    def distance_unit(self) -> Literal['px', 'cm', 'deg']:
        return self._distance_unit

    @distance_unit.setter
    def distance_unit(self, new_unit: Literal['px', 'cm', 'deg']):
        if self._distance_unit == new_unit:
            return
        if new_unit not in ['px', 'cm', 'deg']:
            raise ValueError(f"Invalid distance unit: {new_unit}. Must be one of ['px', 'cm', 'deg'].")
        # convert target distances:
        target_distance_columns = [col for col in self._gaze.columns if col.startswith(cnfg.TARGET_STR)]
        new_distances = self._gaze[target_distance_columns].map(lambda dist: hlp.convert_units(
            dist, self.distance_unit, new_unit, cnfg.TOBII_PIXEL_SIZE_MM / 10, self._subject.screen_distance_cm
        ))
        self._gaze.loc[:, target_distance_columns] = new_distances
        self._distance_unit = new_unit

    def get_search_array(self) -> SearchArray:
        return self._search_array

    def get_triggers(self) -> pd.DataFrame:
        return self._triggers

    def get_gaze(self) -> pd.DataFrame:
        return self._gaze

    def get_eye_movements(self, eye: DominantEyeEnum) -> pd.Series:
        if eye == DominantEyeEnum.Left:
            return self._left_events
        if eye == DominantEyeEnum.Right:
            return self._right_events
        raise ValueError(f"Invalid eye: {eye}. Must be either 'left' or 'right'.")

    def get_subject(self) -> "Subject":
        return self._subject

    def get_targets(self) -> pd.DataFrame:
        target_images = self._search_array.targets
        target_df = pd.DataFrame(target_images, index=[f"{cnfg.TARGET_STR}_{i}" for i in range(len(target_images))])
        target_df[cnfg.CATEGORY_STR] = [img.category for img in target_images]
        return target_df

    def calculate_target_distances(self, x: np.ndarray, y: np.ndarray,) -> pd.DataFrame:
        """
        Calculate the distance from each X-Y coordinate to each target in the search array.
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
                dists[i, j] = hlp.distance(
                    (cx, cy), (tx, ty),
                    unit=self._distance_unit,
                    pixel_size_cm=cnfg.TOBII_PIXEL_SIZE_MM / 10,
                    screen_distance_cm=self._subject.screen_distance_cm
                )
        dists = pd.DataFrame(dists, columns=targets.index)
        return dists

    def extract_target_identification(self) -> pd.DataFrame:
        """
        Extracts the time, gaze location and distance from the target for each target identification during the trial.
        If a taget was never identified, the time and distance are set to np.inf and gaze data is set to NaN.
        If a target was identified multiple times, we record all identifications and raise a warning.

        :return: pd.DataFrame indexed by targets (`target_0`, `target_1`, etc.) with the following columns:
            - `time`: time of identification
            - `distance`: distance from the target at the time of identification
            - `left_x`, `left_y`: gaze coordinates in the left eye at the time of identification
            - `right_x`, `right_y`: gaze coordinates in the right eye at the time of identification
            - `left_pupil`, `right_pupil`: pupil size in the left/right eye at the time of identification
            - `'left_label'`, `'right_label'`: eye movement labels in the left/right eye at the time of identification
            - `target_x`, `target_y`: target coordinates
            - `target_category`: target category
        """
        # extract gaze on target identification
        MAX_GAZE_TO_TRIGGER_TIME_DIFF = 10  # in ms
        identification_triggers = self._triggers[
            self._triggers[cnfg.ACTION_STR] == SearchActionTypesEnum.MARK_AND_CONFIRM
        ].reset_index(drop=True)
        gaze_when_ident = self._gaze.loc[hlp.closest_indices(
            self._gaze['time'], identification_triggers['time'], threshold=MAX_GAZE_TO_TRIGGER_TIME_DIFF
        )].reset_index(drop=True)

        # calculate minimal distance from targets
        dists = gaze_when_ident[[col for col in gaze_when_ident.columns if col.startswith(cnfg.TARGET_STR)]].copy()
        closest_target = dists.idxmin(axis=1)
        dists = pd.Series(
            dists.to_numpy()[dists.index, dists.columns.get_indexer(closest_target)],
            name=f"{cnfg.DISTANCE_STR}_{self.distance_unit}"
        )

        # concatenate results
        res = pd.concat([
            identification_triggers[cnfg.TIME_STR],
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
        gaze_x = self._gaze[cnfg.RIGHT_X_STR if self._subject.eye == DominantEyeEnum.Right else cnfg.LEFT_X_STR].values
        gaze_y = self._gaze[cnfg.RIGHT_Y_STR if self._subject.eye == DominantEyeEnum.Right else cnfg.LEFT_Y_STR].values
        dists = self.calculate_target_distances(gaze_x, gaze_y)
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
