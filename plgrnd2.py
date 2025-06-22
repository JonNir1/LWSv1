from typing import Dict

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
# fixs = fixs[fixs["outlier_reasons"].apply(lambda x: len(x) == 0)]  # drop outliers
visits = get_visits(subj, save=False, verbose=True)


## LWS Funnel - fixations
from analysis.lws_funnel import calc_funnel
import plotly.graph_objects as go

fixs_funnel = calc_funnel(fixations=fixs)
funnel_sizes = {k: len(v) for (k, v) in fixs_funnel.items()}
funnel_fig = go.Figure(go.Funnelarea(
    labels=list(funnel_sizes.keys()),
    values=list(funnel_sizes.values()),
    textinfo="label+value",
    title=dict(text="<b>LWS Funnel - Fixations</b>", font=dict(size=20)),
))
funnel_fig.show()


# TODO: make `calc_funnel` work with visits too
# visits["outlier_reasons"] = [[] for _ in range(len(visits))]
visits_funnel = calc_funnel(visits=visits)
