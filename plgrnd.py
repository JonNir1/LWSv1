import os
import copy

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

# process trial fixations
from analysis.extract_fixations import extract_fixations
fixs_df = extract_fixations(trial)


# LWS instance:
#   fixation on target
#   before identification
#   not coinciding with trial end
#   succeeding fixation is not in bottom strip
#   succeeding fixation is not same-target's identification fixation
# TODO: determine if fixation close to target, number proximal fixations
