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
        labels, left_events, right_events = self._detect_eye_movements()
        self._gaze = pd.concat([self._gaze, labels], axis=1)
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

    def get_icons(self) -> pd.DataFrame:
        """
        Extracts every icon in the trial's search array: pixel coordinates, jitter angle, category, image sub-path,
        and whether it is a target.

        Indexed by the stable `icon{i}` identifier (flat row-major position in the array). Columns are prefixed
        `target_` for backwards compatibility with the consumers of `get_targets()`, which is a filter over this.
        """
        icons = self._search_array.icons
        images = [img for _icon_id, img, _is_tgt in icons]
        icon_df = pd.DataFrame(images, index=[icon_id for icon_id, _img, _is_tgt in icons])
        icon_df[cnfg.CATEGORY_STR] = [img.category.name for img in images]
        icon_df = icon_df.rename(columns=lambda col: f"{cnfg.TARGET_STR}_{col}", inplace=False)
        icon_df["is_target"] = [is_tgt for _icon_id, _img, is_tgt in icons]
        return icon_df

    def get_targets(self) -> pd.DataFrame:
        """
        The target subset of `get_icons()`: pixel coordinates, angle, category, and image path, indexed by the same
        stable `icon{i}` identifier.
        """
        icons = self.get_icons()
        targets = icons.loc[icons["is_target"]].drop(columns=["is_target"])
        assert len(targets) == self.num_targets, (
            f"expected {self.num_targets} targets in trial {self.trial_num}, found {len(targets)}"
        )
        return targets

    def get_metadata(self, bad_actions: Sequence[SubjectActionCategoryEnum]) -> pd.Series:
        return pd.Series({
            "trial": self.trial_num,
            "block": self.block_num,
            "trial_category": self.trial_category.name,
            "duration": self.duration,
            "num_targets": self.num_targets,
            "num_distractors": self.num_distractors,
            "num_actions": self.num_actions,
            "bad_actions": bool(np.isin(self.get_actions()[cnfg.ACTION_STR], bad_actions).any()),   # TODO: remove this
            "gaze_coverage": self.gaze_coverage,
        })

    def get_raw_eye_movements(self) -> pd.DataFrame:
        """ Returns a DataFrame summarizing the eye movements detected during the trial. """
        left = self._summarize_events(self._left_events)
        right = self._summarize_events(self._right_events)
        df = pd.concat(
            [left, right],
            keys=[cnfg.LEFT_STR, cnfg.RIGHT_STR],
            names=[cnfg.EYE_STR, cnfg.EVENT_STR], axis=0
        )
        return df

    def process_events(self) -> pd.DataFrame:
        from data_models.preprocess.events import process_trial_events
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            features = self.get_raw_eye_movements()
        events = process_trial_events(features, self.end_time)
        return events

    @staticmethod
    def _summarize_events(events: Sequence) -> pd.DataFrame:
        """
        `peyes.summarize_events`, plus the two endpoint columns it omits.

        `peyes`' `BaseEvent.summary()` reports `center_pixel` but neither `start_pixel` nor `end_pixel`, although
        both exist as properties on the event. For a saccade those two points *are* the geometry - where the
        movement began and where it landed - and amplitude and azimuth give magnitude and direction but not
        position, so a landing site cannot be recovered from the summary alone. Read them off the `Event` objects
        while we still hold them; drop this once upstream exposes them (feature request filed against `peyes`).
        """
        summary = peyes.summarize_events(events)
        endpoints = pd.DataFrame(
            [(*ev.start_pixel, *ev.end_pixel) for ev in events],
            columns=["start_x", "start_y", "end_x", "end_y"], index=summary.index, dtype=float,
        )
        return pd.concat([summary, endpoints], axis=1)

    def _create_search_array(self) -> SearchArray:
        search_array_type = SearchArrayCategoryEnum[_extract_singleton_column(self._gaze, cnfg.CONDITION_STR).upper()]
        search_array_num = int(_extract_singleton_column(self._gaze, "image_num"))
        return SearchArray.from_mat(
            SearchArray.get_path(cnfg.STIMULI_VERSION, search_array_type, search_array_num, "mat")
        )

    def _detect_eye_movements(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        from data_models.parse.eye_movements import detect_eye_movements
        left_labels, left_events = detect_eye_movements(
            self._gaze,
            DominantEyeEnum.LEFT,
            self._subject.screen_distance_cm,
            pixel_size_cm=cnfg.PIXEL_SIZE_CM,
            only_labels=False
        )
        right_labels, right_events = detect_eye_movements(
            self._gaze,
            DominantEyeEnum.RIGHT,
            self._subject.screen_distance_cm,
            pixel_size_cm=cnfg.PIXEL_SIZE_CM,
            only_labels=False
        )
        labels = pd.concat([left_labels, right_labels], axis=1)
        return labels, left_events, right_events

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

    def __hash__(self) -> int:
        """
        Defining `__eq__` without this sets `__hash__ = None`, making `Trial` unhashable - so it cannot go in a set
        or be a dict key, which is surprising for a value-like object.

        Hashes a subset of the fields `__eq__` compares. That is the required contract: equal trials agree on every
        comparison field, so they agree on this subset too. A `Trial` is effectively immutable after `__init__`.
        """
        return hash((self.block_num, self.trial_num, self.start_time, self.end_time))

    def __repr__(self) -> str:
        return f"Trial {self.trial_num} ({self.trial_category.name})"
