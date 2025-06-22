import numpy as np
import pandas as pd
import plotly.io as pio

import config as cnfg
from data_models.LWSEnums import DominantEyeEnum
from data_models.Subject import Subject

from analysis.fixations import get_fixations
from analysis.visits import get_visits

pio.renderers.default = "browser"


## Read Subject Data
subj = Subject.from_raw(
    exp_name=cnfg.EXPERIMENT_NAME, subject_id=1, session=1, data_dir="v4-1-1 GalChen Demo", verbose=True
)
# subj.to_pickle(overwrite=False)
# subj = Subject.from_pickle(exp_name=cnfg.EXPERIMENT_NAME, subject_id=1,)

idents = subj.get_target_identification_summary()
fixs = get_fixations(subj, save=False, verbose=True)
fixs = fixs[fixs["outlier_reasons"].apply(lambda x: len(x) == 0)]  # drop outliers
visits = get_visits(subj, save=False, verbose=True)
