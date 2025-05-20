import os
import copy

import numpy as np
import pandas as pd
import peyes

import config as cnfg
from data_models.LWSEnums import DominantEyeEnum, SearchActionTypesEnum
from data_models.Subject import Subject

SUBJ_PREFIX = "v4-1-1"
SUBJ_PATH = os.path.join(cnfg.RAW_DATA_PATH, f"{SUBJ_PREFIX} GalChen Demo")


# read subject data
subj = Subject.from_raw(
    exp_name=cnfg.EXPERIMENT_NAME, subject_id=1, session=1, dirname="v4-1-1 GalChen Demo", verbose=True
)

# extract specific trial
trial = subj.get_trials()[33]

arr = trial.get_search_array()
targets = trial.get_targets()
triggers = trial.get_triggers()
gaze = trial.get_gaze()

# identify targets' identification time
import helpers as hlp


target_identification_data = trial.extract_target_identification()
# TODO: ignore if distance is above predefined threshold (replace `time>threshold` with inf)










# detect eye movements
from parse.eye_movements import detect_eye_movements
labels, events = detect_eye_movements(
    gaze, subj.eye, subj.screen_distance_cm, cnfg.DETECTOR, cnfg.TOBII_PIXEL_SIZE_MM / 10, only_labels=False
)
events_df = peyes.summarize_events(events)

fixations = list(filter(lambda evnt: evnt.label == 1, events))
fix1 = copy.deepcopy(fixations[0])
fix1.distance_to_t1 = 100
