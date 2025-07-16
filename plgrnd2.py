import numpy as np
import pandas as pd
import plotly.io as pio
from plotly.subplots import make_subplots

import config as cnfg
from data_models.LWSEnums import SignalDetectionCategoryEnum

pio.renderers.default = "browser"


# %%
# ##  Run Pipeline / Load Data
# from analysis.pipeline.full_pipeline import full_pipeline
# targets, actions, metadata, idents, fixations, visits = full_pipeline(      # uncomment to re-run
#     on_target_threshold_dva=cnfg.ON_TARGET_THRESHOLD_DVA,
#     visit_merging_time_threshold=cnfg.VISIT_MERGING_TIME_THRESHOLD,
#     save=True, verbose=True
# )
from analysis.pipeline.full_pipeline import read_saved_data
targets, actions, metadata, idents, fixations, visits = read_saved_data()


# %%
# ## Detect LWS Instances
from analysis.helpers.funnels import calc_funnel_sizes
from analysis.figures.lws.funnel_fig import create_funnel_figure

create_funnel_figure(
    calc_funnel_sizes(visits, "target_return"), "target_return", "visits", show_individuals=True
).show()



# %%
from analysis.helpers.sdt import calc_sdt_metrics
from analysis.figures.subject_comparisons.signal_detection import signal_detection_figure
signal_detection_figure(calc_sdt_metrics(metadata, idents, "loglinear")).show()

from analysis.figures.subject_comparisons.identifications import identifications_figure
identifications_figure(idents, metadata).show()






# %%


# TODO: timings
#  - from trial start to first action (including bad actions)
#  - from last action (including bad actions) to trial end
#  - from trial start to first identification (hit)
#  - from last identification (hit) to trial end
#  - from trial start to first fixation/visit on target
#  - from last fixation/visit on target to trial end


fig = make_subplots(
    rows=1, cols=2,
)
bad_actions_count = metadata.groupby(cnfg.SUBJECT_STR)["bad_actions"].sum().reset_index()

# find which subject-trial combinations have no actions
subjects = actions[cnfg.SUBJECT_STR].unique()
trials = np.arange(1, 61)  # assuming trials are numbered from 1 to 60
expected = pd.MultiIndex.from_product([subjects, trials], names=[cnfg.SUBJECT_STR, cnfg.TRIAL_STR])
observed = actions.set_index([cnfg.SUBJECT_STR, cnfg.TRIAL_STR]).index.drop_duplicates()
missing = expected.difference(observed).to_frame(index=False)
missing_actions_count = missing.groupby(cnfg.SUBJECT_STR).size().rename("missing_actions").reset_index()
del subjects, trials, expected, observed, missing

visits_td = dict()
for (s, t, e, tgt), data in visits.groupby(["subject", "trial", "eye", "target"]):
    visits_td[(s, t, e, tgt)] = (data["start_time"] - data["end_time"].shift(1)).rename("time_diff").dropna()
visits_td = (
    pd.concat(visits_td.values(), keys=visits_td.keys(), names=["subject", "trial", "eye", "target"])
    .droplevel(-1)
    .reset_index()
)



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
