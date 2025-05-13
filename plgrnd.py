import os

import numpy as np
import pandas as pd
import peyes

import config as cnfg
from data_models.LWSEnums import DominantEyeEnum
from data_models.Subject import Subject

SUBJ_PREFIX = "v4-1-1"
SUBJ_PATH = os.path.join(cnfg.RAW_DATA_PATH, f"{SUBJ_PREFIX} GalChen Demo")


# read subject data
subj = Subject.from_raw(
    exp_name=cnfg.EXPERIMENT_NAME, subject_id=1, session=1, dirname="v4-1-1 GalChen Demo", verbose=True
)

# extract specific trial
trial = subj.get_trials()[9]
arr = trial.get_search_array()

# detect eye movements
detector = peyes.create_detector('Engbert', cnfg.MISSING_VALUE, 5, 0)

trial_gaze = trial.get_gaze()
t = trial_gaze[cnfg.TIME_STR].values
x = trial_gaze[cnfg.RIGHT_X_STR if subj.hand == DominantEyeEnum.Right else cnfg.LEFT_X_STR].values
y = trial_gaze[cnfg.RIGHT_Y_STR if subj.hand == DominantEyeEnum.Right else cnfg.LEFT_Y_STR].values
pupil = trial_gaze[cnfg.RIGHT_PUPIL_STR if subj.hand == DominantEyeEnum.Right else cnfg.LEFT_PUPIL_STR].values

labels, _ = detector.detect(
    t=t, x=x, y=y, viewer_distance_cm=subj.screen_distance_cm, pixel_size_cm=cnfg.TOBII_PIXEL_SIZE_MM / 10,
)
events = peyes.create_events(
    labels=labels, t=t, x=x, y=y, pupil=pupil,
    viewer_distance=subj.screen_distance_cm, pixel_size=cnfg.TOBII_PIXEL_SIZE_MM / 10
)
events_df = peyes.summarize_events(events)
del t, x, y, pupil

