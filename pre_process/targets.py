import pandas as pd
from tqdm import tqdm

import config as cnfg
from data_models.Subject import Subject


def extract_targets(subject: Subject):
    """
    Extract the target information from a subject's trials, returning a DataFrame with the following columns:
    - trial: int; the trial number
    - target: str; the name of the target
    - x: float; the x coordinate of the target in pixels
    - y: float; the y coordinate of the target in pixels
    - angle: float; the rotation angle of the target in degrees
    - category: ImageCategoryEnum; the category of the target
    - sub_path: str; the path to the target image file, relative to the images directory
    """
    targets = {
        trial.trial_num: (
            trial.get_targets()
            .rename(columns=lambda name: name.replace(f"{cnfg.TARGET_STR}_", ""))
            .reset_index(drop=False)
            .rename(columns={"index": cnfg.TARGET_STR,})
            .sort_values(by=cnfg.TARGET_STR)
        ) for trial in tqdm(subject.get_trials(), desc="Extracting Targets", disable=True)
    }
    targets = (
        pd.concat(targets.values(), axis=0, keys=targets.keys())
        .reset_index(drop=False)
        .rename(columns={"level_0": cnfg.TRIAL_STR})
        .drop(columns=["level_1"])
    )
    return targets
