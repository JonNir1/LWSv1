from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cnfg
from data_models.Subject import Subject
from data_models.Trial import Trial
from data_models.LWSEnums import SubjectActionTypesEnum


def extract_metadata(
        subject: Subject, bad_actions: Sequence[SubjectActionTypesEnum]
) -> pd.DataFrame:
    """
    Extract the subject's trial metadata into a DataFrame, containing the following columns:
    - trial_num
    - block_num
    - trial_type: COLOR/BW/NOISE
    - duration: in ms
    - num_targets
    - bad_actions: boolean; True if any of the subject's actions during the trial are considered bad
    """
    dfs = {
        trial.trial_num: extract_trial_metadata(trial, bad_actions)
        for trial in tqdm(subject.get_trials(), desc="Trial Metadata", disable=True)
    }
    res = pd.concat(dfs.values(), keys=dfs.keys(), axis=1,).T
    res = res.reset_index(drop=True)
    return res


def extract_trial_metadata(
        trial: Trial, bad_actions: Sequence[SubjectActionTypesEnum]
) -> pd.Series:
    return pd.Series({
        "trial_num": trial.trial_num,
        "block_num": trial.block_num,
        "trial_type": trial.trial_type.name,
        "duration": trial.end_time - trial.start_time,
        "num_targets": len(trial.get_search_array().targets),
        f"bad_actions": bool(np.isin(trial.get_actions()[cnfg.ACTION_STR], bad_actions).any()),
    })

