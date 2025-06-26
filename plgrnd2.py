from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
subj.to_pickle(overwrite=False)
subj = Subject.from_pickle(exp_name=cnfg.EXPERIMENT_NAME, subject_id=1,)

idents = subj.get_target_identification_summary()
fixs = get_fixations(subj, save=True, verbose=True)
# fixs = fixs[fixs["outlier_reasons"].apply(lambda x: len(x) == 0)]  # drop outliers
visits = get_visits(subj, save=True, verbose=True)


## LWS Funnel - fixations
from analysis.lws_funnel import fixation_funnel
fixs_funnel = fixation_funnel(fixations=fixs)
fix_funnel_sizes = {k: len(v) for (k, v) in fixs_funnel.items()}
fix_funnel_fig = go.Figure(go.Funnelarea(
    labels=list(fix_funnel_sizes.keys()),
    values=list(fix_funnel_sizes.values()),
    textinfo="label+value",
    title=dict(text="<b>LWS Funnel - Fixations</b>", font=dict(size=20)),
))
fix_funnel_fig.show()
lws_fixations = fixs_funnel["not_end_with_trial"]

## LWS Funnel - visits
from analysis.lws_funnel import visit_funnel
visits_funnel = visit_funnel(visits=visits, distance_col="weighted_distance")
vis_funnel_sizes = {k: len(v) for (k, v) in visits_funnel.items()}
vis_funnel_fig = go.Figure(go.Funnelarea(
    labels=list(vis_funnel_sizes.keys()),
    values=list(vis_funnel_sizes.values()),
    textinfo="label+value",
    title=dict(text="<b>LWS Funnel - Visits</b>", font=dict(size=20)),
))
vis_funnel_fig.show()
lws_visits = visits_funnel["not_end_with_trial"]
