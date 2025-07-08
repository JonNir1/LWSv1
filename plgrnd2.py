
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

import config as cnfg
from analysis.pipeline.full_pipeline import full_pipeline

pio.renderers.default = "browser"


targets, metadata, idents, fixations, visits = full_pipeline(verbose=True)


## LWS Funnel - fixations/visits
from analysis.figures.funnel_fig import create_funnel_figure
create_funnel_figure(visits, "visits", show_individuals=True).show()

## timings
idents_hit_data = idents.loc[
    idents[f"{cnfg.DISTANCE_STR}_dva"] <= cnfg.ON_TARGET_THRESHOLD_DVA,
    [cnfg.SUBJECT_STR, cnfg.TRIAL_STR, cnfg.TIME_STR, f"{cnfg.DISTANCE_STR}_dva"]
].copy().map(lambda val: round(val, 2))

fig = make_subplots(
    rows=2, cols=3, shared_xaxes=False, shared_yaxes=False,
)


