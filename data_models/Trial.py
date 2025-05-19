from __future__ import annotations
import os
from typing import Optional, Literal

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

        # generate the search array
        search_array_type = SearchArrayTypeEnum[_extract_singleton_column(gaze, cnfg.CONDITION_STR).upper()]
        search_array_num = int(_extract_singleton_column(gaze, "image_num"))
        self._search_array = SearchArray.from_mat(os.path.join(
            cnfg.SEARCH_ARRAY_PATH,
            f"generated_stim{cnfg.STIMULI_VERSION}",
            search_array_type.name.lower(),
            f"image_{search_array_num}.mat"
        ))

        # process gaze data
        labels = self.detect_eye_movements()
        dists = self.calculate_gaze_target_distances(unit=self._distance_unit)
        self._gaze = pd.concat([self._gaze, labels, dists], axis=1)

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
    def distance_unit(self, value: Literal['px', 'cm', 'deg']):
        if self._distance_unit == value:
            return
        if value not in ['px', 'cm', 'deg']:
            raise ValueError(f"Invalid distance unit: {value}. Must be one of ['px', 'cm', 'deg'].")
        # recalculate target distances:
        dists = self.calculate_gaze_target_distances(unit=value)
        self._gaze.loc[:, dists.columns] = dists
        self._distance_unit = value

    def get_search_array(self) -> SearchArray:
        return self._search_array

    def get_triggers(self) -> pd.DataFrame:
        return self._triggers

    def get_gaze(self) -> pd.DataFrame:
        return self._gaze

    def get_subject(self) -> "Subject":
        return self._subject

    def get_targets(self) -> pd.DataFrame:
        target_images = self._search_array.targets
        target_df = pd.DataFrame(target_images, index=[f"{cnfg.TARGET_STR}_{i}" for i in range(len(target_images))])
        target_df[cnfg.CATEGORY_STR] = [img.category for img in target_images]

        # extract when targets were identified by the subject
        # target_df["time_identified"] = np.inf
        return target_df

    def detect_eye_movements(self):
        if cnfg.LEFT_LABEL_STR in self._gaze.columns:
            left_labels = self._gaze[cnfg.LEFT_LABEL_STR]
        else:
            left_labels = detect_eye_movements(
                self._gaze, DominantEyeEnum.Left, self._subject.screen_distance_cm, cnfg.DETECTOR, cnfg.TOBII_PIXEL_SIZE_MM / 10
            )
        if cnfg.RIGHT_LABEL_STR in self._gaze.columns:
            right_labels = self._gaze[cnfg.RIGHT_LABEL_STR]
        else:
            right_labels = detect_eye_movements(
                self._gaze, DominantEyeEnum.Right, self._subject.screen_distance_cm, cnfg.DETECTOR, cnfg.TOBII_PIXEL_SIZE_MM / 10
            )
        labels = pd.concat([left_labels, right_labels], axis=1)
        return labels

    def calculate_gaze_target_distances(self, unit: Optional[Literal['px', 'cm', 'deg']] = None) -> pd.DataFrame:
        gaze_x_col = cnfg.RIGHT_X_STR if self._subject.eye == DominantEyeEnum.Right else cnfg.LEFT_X_STR
        gaze_y_col = cnfg.RIGHT_Y_STR if self._subject.eye == DominantEyeEnum.Right else cnfg.LEFT_Y_STR
        gaze_coords = self._gaze[[gaze_x_col, gaze_y_col]].values  # shape (n_gazes, 2)
        targets = self.get_targets()
        target_coords = targets[[cnfg.X, cnfg.Y]].values  # shape (n_targets, 2)
        dists = np.empty((len(gaze_coords), len(targets)), dtype=float)
        for i, (gx, gy) in enumerate(gaze_coords):
            for j, (tx, ty) in enumerate(target_coords):
                dists[i, j] = hlp.distance(
                    (gx, gy), (tx, ty),
                    unit=self._distance_unit if unit is None else unit,
                    pixel_size_cm=cnfg.TOBII_PIXEL_SIZE_MM / 10,
                    screen_distance_cm=self._subject.screen_distance_cm
                )
        dists = pd.DataFrame(dists, index=self._gaze.index, columns=targets.index)
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
