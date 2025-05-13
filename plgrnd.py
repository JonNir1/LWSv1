import os

import numpy as np
import pandas as pd
import peyes

import config as cnfg
from data_models.LWSEnums import SearchArrayTypeEnum, ImageCategoryEnum, DominantEyeEnum
from data_models.Subject import Subject
from data_models.SearchArray import SearchArray

STIMULI_VERSION = 1
SUBJ_PREFIX = "v4-1-1"
SUBJ_PATH = os.path.join(cnfg.RAW_DATA_PATH, f"{SUBJ_PREFIX} GalChen Demo")


# read subject data
subj = Subject.from_raw(exp_name=cnfg.EXPERIMENT_NAME, subject_id=1, session=1, dirname="v4-1-1 GalChen Demo")
data = subj.get_behavior()

# extract single trial data
trial = data[data[cnfg.TRIAL_STR] == 10].copy(deep=True)
trial = trial.iloc[2:]  # first 2 rows are too early    # TODO: find a way to remove these globally

array_type = SearchArrayTypeEnum[trial[cnfg.CONDITION_STR][trial[cnfg.CONDITION_STR].notnull()].iloc[0].upper()]
image_num = int(trial[f"image_num"][trial[f"image_num"].notnull()].iloc[0])

search_array = SearchArray.from_mat(os.path.join(
    cnfg.SEARCH_ARRAY_PATH,
    f"generated_stim{STIMULI_VERSION}",
    array_type.name.lower(),
    f"image_{image_num}.mat"
))
del array_type, image_num

# detect eye movements
detector = peyes.create_detector('Engbert', cnfg.MISSING_VALUE, 5, 0)

t = trial[cnfg.TIME_STR].values
x = trial[cnfg.RIGHT_X_STR if subj.hand == DominantEyeEnum.Right else cnfg.LEFT_X_STR].values
x = x * cnfg.TOBII_MONITOR.width     # correct to tobii's resolution
y = trial[cnfg.RIGHT_Y_STR if subj.hand == DominantEyeEnum.Right else cnfg.LEFT_Y_STR].values
y = y * cnfg.TOBII_MONITOR.height    # correct to tobii's resolution
pupil = trial[cnfg.RIGHT_PUPIL_STR if subj.hand == DominantEyeEnum.Right else cnfg.LEFT_PUPIL_STR].values

labels, _ = detector.detect(
    t=t, x=x, y=y, viewer_distance_cm=subj.screen_distance_cm, pixel_size_cm=cnfg.TOBII_PIXEL_SIZE_MM / 10,
)
events = peyes.create_events(
    labels=labels, t=t, x=x, y=y, pupil=pupil,
    viewer_distance=subj.screen_distance_cm, pixel_size=cnfg.TOBII_PIXEL_SIZE_MM / 10
)
events_df = peyes.summarize_events(events)
del t, x, y, pupil

fixations = events_df[events_df["label"] == 1]

