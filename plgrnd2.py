from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

import config as cnfg
from data_models.LWSEnums import SignalDetectionCategoryEnum

pio.renderers.default = "browser"


##  Run Pipeline / Load Data
from analysis.pipeline.full_pipeline import full_pipeline, read_saved_data
# targets, actions, metadata, idents, fixations, visits = full_pipeline(      # uncomment to re-run
#     on_target_threshold_dva=cnfg.ON_TARGET_THRESHOLD_DVA,
#     visit_merging_time_threshold=cnfg.VISIT_MERGING_TIME_THRESHOLD,
#     save=True, verbose=True
# )
targets, actions, metadata, idents, fixations, visits = read_saved_data()





idents_hit_data = (
    idents
    .copy()
    .loc[
        idents[cnfg.IDENTIFICATION_CATEGORY_STR] == SignalDetectionCategoryEnum.HIT,
        [cnfg.SUBJECT_STR, cnfg.TRIAL_STR, cnfg.TARGET_STR, cnfg.IDENTIFICATION_CATEGORY_STR, cnfg.TIME_STR, cnfg.DISTANCE_DVA_STR]
    ]
    .map(lambda val: round(val, 2) if isinstance(val, float) else val)
    .merge(
        targets[[cnfg.SUBJECT_STR, cnfg.TRIAL_STR, cnfg.TARGET_STR, cnfg.CATEGORY_STR]],
        on=[cnfg.SUBJECT_STR, cnfg.TRIAL_STR, cnfg.TARGET_STR],
        how='left'
    )
)


fig = make_subplots(
    rows=2, cols=3, shared_xaxes=False, shared_yaxes=False,
)

# %% ## Plot Subject-Performance Comparison by Trial Category
from analysis.figures.performance_outliers import create_subject_comparison_figure
create_subject_comparison_figure(metadata, idents).show()


# %% ## Detect LWS Instances
from analysis.lws_funnel import fixation_funnel, visit_funnel, calc_funnel_sizes
from analysis.figures.funnel_fig import create_funnel_figure
lws_fixation_funnel = fixation_funnel(
    fixations, metadata, idents,
    on_target_threshold_dva=cnfg.ON_TARGET_THRESHOLD_DVA,
    fixs_to_strip_threshold=cnfg.FIXATIONS_TO_STRIP_THRESHOLD,
    time_to_trial_end_threshold=cnfg.TIME_TO_TRIAL_END_THRESHOLD
)
lws_fixation_funnel_sizes = calc_funnel_sizes(lws_fixation_funnel)
create_funnel_figure(lws_fixation_funnel_sizes, "fixations", show_individuals=True).show()
del lws_fixation_funnel, lws_fixation_funnel_sizes

lws_visit_funnel = visit_funnel(
    visits, metadata, idents,
    on_target_threshold_dva=cnfg.ON_TARGET_THRESHOLD_DVA,
    fixs_to_strip_threshold=cnfg.FIXATIONS_TO_STRIP_THRESHOLD,
    time_to_trial_end_threshold=cnfg.TIME_TO_TRIAL_END_THRESHOLD,
    distance_type='min'     # can also be 'max' or 'weighted'
)
lws_visit_funnel_sizes = calc_funnel_sizes(lws_visit_funnel)
create_funnel_figure(lws_visit_funnel_sizes, "visits", show_individuals=True).show()
del lws_visit_funnel, lws_visit_funnel_sizes

