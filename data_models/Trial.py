import os

import numpy as np
import pandas as pd

import config as cnfg
from data_models.SearchArray import SearchArray
from data_models.LWSEnums import SearchArrayTypeEnum


class Trial:
    """
    A class to represent a single LWS trial.
    Each trial consists of its SearchArray and behavioral data (pd.DataFrame).
    """

    def __init__(
            self, block_num: int, trial_num: int, behavior: pd.DataFrame, search_array: SearchArray,
    ):
        self._block_num = block_num
        self._trial_num = trial_num
        self._behavior = behavior
        self._search_array = search_array

        # verify inputs match:
        behavior_block_num = int(Trial.__extract_singleton_column(behavior, cnfg.BLOCK_STR))
        assert block_num == behavior_block_num, f"Block num {block_num} does not match behavior's block num {behavior_block_num}."
        behavior_trial_num = int(Trial.__extract_singleton_column(behavior, cnfg.TRIAL_STR))
        assert trial_num == behavior_trial_num, f"Trial num {trial_num} does not match behavior's trial num {behavior_trial_num}."
        behavior_search_array_type = SearchArrayTypeEnum[Trial.__extract_singleton_column(behavior, cnfg.CONDITION_STR).upper()]
        assert search_array.array_type == behavior_search_array_type, f"Array type {search_array.array_type.name} does not match behavior's array type {behavior_search_array_type.name}."
        behavior_search_array_num = int(Trial.__extract_singleton_column(behavior, "image_num"))
        assert search_array.image_num == behavior_search_array_num, f"Array num {search_array.image_num} does not match behavior's array num {behavior_search_array_num}."

    @staticmethod
    def from_behavior(behavior: pd.DataFrame) -> "Trial":
        # extract block and trial numbers
        block_num = int(Trial.__extract_singleton_column(behavior, cnfg.BLOCK_STR))
        trial_num = int(Trial.__extract_singleton_column(behavior, cnfg.TRIAL_STR))

        # generate the search array
        search_array_type = SearchArrayTypeEnum[Trial.__extract_singleton_column(behavior, cnfg.CONDITION_STR).upper()]
        search_array_num = int(Trial.__extract_singleton_column(behavior, "image_num"))
        search_array = SearchArray.from_mat(os.path.join(
            cnfg.SEARCH_ARRAY_PATH,
            f"generated_stim{cnfg.STIMULI_VERSION}",
            search_array_type.name.lower(),
            f"image_{search_array_num}.mat"
        ))
        return Trial(block_num, trial_num, behavior, search_array)


    @property
    def block_num(self) -> int:
        return self._block_num

    @property
    def trial_num(self) -> int:
        return self._trial_num

    def get_behavior(self) -> pd.DataFrame:
        return self._behavior

    def get_search_array(self) -> SearchArray:
        return self._search_array

    def __eq__(self, other) -> bool:
        if not isinstance(other, Trial):
            return False
        if self.trial_num != other.trial_num:
            return False
        if self.block_num != other.block_num:
            return False
        if not self.get_behavior().equals(other.get_behavior()):
            return False
        if self.get_search_array() != other.get_search_array():
            return False
        return True

    @staticmethod
    def __extract_singleton_column(behavior: pd.DataFrame, col_name: str):
        values = behavior[col_name].dropna()
        assert values.nunique() == 1, f"Behavioral data contains multiple values in column {col_name}"
        return values.iloc[0]
