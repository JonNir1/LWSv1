import os

import numpy as np
import pandas as pd

import config as cnfg
import parse_data as prsr
from data_models.Subject import Subject

SUBJ_PREFIX = "v4-1-1"
SUBJ_PATH = os.path.join(cnfg.RAW_DATA_PATH, f"{SUBJ_PREFIX} GalChen Demo")

gaze = prsr._read_gaze(os.path.join(SUBJ_PATH, f"{SUBJ_PREFIX}-GazeData.txt"))
# triggers = prsr.read_triggers(os.path.join(SUBJ_PATH, f"{SUBJ_PREFIX}-TriggerLog.txt"))

behavior = prsr.parse_behavioral_data(
    os.path.join(SUBJ_PATH, f"{SUBJ_PREFIX}-TriggerLog.txt"),
    os.path.join(SUBJ_PATH, f"{SUBJ_PREFIX}-GazeData.txt")
)


subj = Subject.from_raw(exp_name=cnfg.EXPERIMENT_NAME, subject_id=1, session=1, dirname="v4-1-1 GalChen Demo")
data = subj.get_behavior()



