from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

import config as cnfg
from data_models.LWSEnums import SubjectActionTypesEnum
from analysis.pipeline.full_pipeline import full_pipeline

pio.renderers.default = "browser"


targets, metadata, idents, fixations, visits = full_pipeline(verbose=True)


## LWS Funnel - fixations/visits
from analysis.figures.funnel_fig import create_funnel_figure
create_funnel_figure(visits, "visits", show_individuals=True).show()

## trial validity: number of trials with bad actions / false alrams, by subject and trial type

from analysis.trial_type import calc_bad_actions_rate, calc_sdt_class_rate, calc_dprime
bad_actions_rate = calc_bad_actions_rate(metadata)
hit_rate = calc_sdt_class_rate(metadata, idents, "hit", cnfg.ON_TARGET_THRESHOLD_DVA)
fa_rate = calc_sdt_class_rate(metadata, idents, "false_alarm", cnfg.ON_TARGET_THRESHOLD_DVA)
d_prime = calc_dprime(metadata, idents, cnfg.ON_TARGET_THRESHOLD_DVA)


# TODO: move to its own figure file
fig = make_subplots(
    rows=1, cols=2, shared_xaxes=True, shared_yaxes=True,
)
fig.add_trace(
    row=1, col=1,
    trace=go.Violin(
        x=bad_actions_rate[cnfg.TRIAL_TYPE_STR],
        y=bad_actions_rate["mean"],
        box_visible=False, showlegend=False,
        points="all", pointpos=-1.8,
        spanmode='hard',
        name="Bad Actions Rate",
    )
)
fig.add_trace(
    row=1, col=2,
    trace=go.Violin(
        x=d_prime[cnfg.TRIAL_TYPE_STR],
        y=d_prime["mean"],
        box_visible=False, showlegend=False,
        points="all", pointpos=-1.8,
        spanmode='hard',
        name="d' (d-prime)",
    )
)
fig.show()




# TODO: timings
#  - from trial start to identification
#  - from trial start to first fixation/visit on target
#  - from last fixation/visit on target to trial end
#  - from last action (including bad actions) to trial end


idents_hit_data = idents.loc[
    idents[f"{cnfg.DISTANCE_STR}_dva"] <= cnfg.ON_TARGET_THRESHOLD_DVA,
    [cnfg.SUBJECT_STR, cnfg.TRIAL_STR, cnfg.TIME_STR, f"{cnfg.DISTANCE_STR}_dva"]
].copy().map(lambda val: round(val, 2))

fig = make_subplots(
    rows=2, cols=3, shared_xaxes=False, shared_yaxes=False,
)


