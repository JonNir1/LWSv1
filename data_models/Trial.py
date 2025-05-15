import os

import numpy as np
import pandas as pd

import config as cnfg
from data_models.SearchArray import SearchArray
from data_models.LWSEnums import SearchArrayTypeEnum

_KEYBOARD_TRIGGERS = [
    cnfg.ExperimentTriggerEnum.SPACE_ACT, cnfg.ExperimentTriggerEnum.SPACE_NO_ACT,
    cnfg.ExperimentTriggerEnum.CONFIRM_ACT, cnfg.ExperimentTriggerEnum.CONFIRM_NO_ACT,
    cnfg.ExperimentTriggerEnum.NOT_CONFIRM_ACT, cnfg.ExperimentTriggerEnum.NOT_CONFIRM_NO_ACT,
    # cnfg.ExperimentTriggerEnum.OTHER_KEY, cnfg.ExperimentTriggerEnum.ABORT_TRIAL,
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

    def __init__(
            self, block_num: int, trial_num: int, search_array: SearchArray, triggers: pd.DataFrame, gaze: pd.DataFrame,
    ):
        self._block_num = block_num
        self._trial_num = trial_num
        self._search_array = search_array
        self._triggers = triggers
        self._gaze = gaze
        self.__validate_inputs()

    @staticmethod
    def from_frames(triggers: pd.DataFrame, gaze: pd.DataFrame) -> "Trial":
        # extract block and trial numbers
        block_num = int(_extract_singleton_column(gaze, cnfg.BLOCK_STR))
        trial_num = int(_extract_singleton_column(gaze, cnfg.TRIAL_STR))

        # generate the search array
        search_array_type = SearchArrayTypeEnum[_extract_singleton_column(gaze, cnfg.CONDITION_STR).upper()]
        search_array_num = int(_extract_singleton_column(gaze, "image_num"))
        search_array = SearchArray.from_mat(os.path.join(
            cnfg.SEARCH_ARRAY_PATH,
            f"generated_stim{cnfg.STIMULI_VERSION}",
            search_array_type.name.lower(),
            f"image_{search_array_num}.mat"
        ))
        return Trial(block_num, trial_num, search_array, triggers, gaze)


    @property
    def block_num(self) -> int:
        return self._block_num

    @property
    def trial_num(self) -> int:
        return self._trial_num

    def get_search_array(self) -> SearchArray:
        return self._search_array

    def get_triggers(self) -> pd.DataFrame:
        return self._triggers

    def get_gaze(self) -> pd.DataFrame:
        return self._gaze

    def __validate_inputs(self):
        # block number
        triggers_block_num = int(_extract_singleton_column(self._triggers, cnfg.BLOCK_STR))
        gaze_block_num = int(_extract_singleton_column(self._gaze, cnfg.BLOCK_STR))
        assert self._block_num == triggers_block_num, f"Block num {self._block_num} does not match triggers block num {triggers_block_num}."
        assert self._block_num == gaze_block_num, f"Block num {self._block_num} does not match gaze block num {gaze_block_num}."
        # trial number
        triggers_trial_num = int(_extract_singleton_column(self._triggers, cnfg.TRIAL_STR))
        gaze_trial_num = int(_extract_singleton_column(self._gaze, cnfg.TRIAL_STR))
        assert self.trial_num == triggers_trial_num, f"Trial num {self.trial_num} does not match triggers trial num {triggers_trial_num}."
        assert self.trial_num == gaze_trial_num, f"Trial num {self.trial_num} does not match gaze trial num {gaze_trial_num}."
        # search array type
        gaze_search_array_type = SearchArrayTypeEnum[_extract_singleton_column(self._gaze, cnfg.CONDITION_STR).upper()]
        assert self._search_array.array_type == gaze_search_array_type, f"Array type {self._search_array.array_type.name} does not match gaze array type {gaze_search_array_type.name}."
        # search array number
        gaze_search_array_num = int(_extract_singleton_column(self._gaze, "image_num"))
        assert self._search_array.image_num == gaze_search_array_num, f"Array num {self._search_array.image_num} does not match behavior's array num {gaze_search_array_num}."

    def __eq__(self, other) -> bool:
        if not isinstance(other, Trial):
            return False
        if self.trial_num != other.trial_num:
            return False
        if self.block_num != other.block_num:
            return False
        if self.get_search_array() != other.get_search_array():
            return False
        return True

    def __repr__(self) -> str:
        return f"Trial {self.trial_num} ({self.get_search_array().array_type.name})"
