import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import config as cnfg

pio.renderers.default = "browser"

# TODO: only from HOME:
cnfg.OUTPUT_PATH = r'C:\Users\nirjo\Desktop\LWS\Results'


# %%
# ##  Run Pipeline / Load Data
# from preprocess.pipeline import full_pipeline
# targets, actions, metadata, idents, fixations, visits = full_pipeline(
#     # raw_data_path=cnfg.RAW_DATA_PATH,
#     # identification_actions=cnfg.IDENTIFICATION_ACTIONS,
#     # on_target_threshold_dva=cnfg.ON_TARGET_THRESHOLD_DVA,
#     # gaze_to_trigger_time_threshold=cnfg.MAX_GAZE_TO_TRIGGER_TIME_DIFF,
#     # visit_merging_time_threshold=cnfg.VISIT_MERGING_TIME_THRESHOLD,
#     save=True,
#     verbose=True
# )

from preprocess.read_data import read_saved_data
targets, actions, metadata, idents, fixations, visits = read_saved_data(cnfg.OUTPUT_PATH)


# %%
from analysis.funnel.prepare import prepare_funnel
from analysis.funnel.prepare import get_funnel_steps
from analysis.funnel.proportion import calculate_funnel_sizes, calculate_proportions

initial_step = "instance_on_target"

funnel_data = prepare_funnel(
    data_dir=cnfg.OUTPUT_PATH,
    funnel_type="lws",
    event_type="visit",
    verbose=True,
)

sizes = calculate_funnel_sizes(funnel_data, get_funnel_steps("lws"), verbose=True)

prop_by_trial = calculate_proportions(
    sizes,
    nominator="final",
    denominator=initial_step,
    aggregate_by="trial_category",
    per_subject=True,
)

prop_by_target = calculate_proportions(
    sizes,
    nominator="final",
    denominator=initial_step,
    aggregate_by="target_category",
    per_subject=True,
)


# %%
from analysis.funnel.visualizations.step_size import step_sizes_figure

step_sizes_figure(
    funnel_data, initial_step, "final", "LWS Visits Funnel", show_individuals=True
).show()

# %%
from analysis.funnel.visualizations.category_comparison import category_comparison_figure

fig = category_comparison_figure(
    prop_by_trial,
    categ_col="trial_category",
    title="LWS Visit Funnel Proportions by Trial Category",
    show_distributions=True,
    show_individuals=True,
)


# %%

# TODO: compare LWS/target-return proportions across trial types & target types

# TODO: pipeline hyperparameter tuning for eye tracking hyperparameters


# TODO: timings
#  - from trial start to first action (including bad actions)
#  - from last action (including bad actions) to trial end
#  - from trial start to first identification (hit)
#  - from last identification (hit) to trial end
#  - from trial start to first fixation/visit on target
#  - from last fixation/visit on target to trial end

# TODO:
#   fixation duration + count distribution
#   saccade duration + amplitude + count distribution
#   micro-saccade duration + amplitude + count distribution

# TODO:
#   fixation duration within trial-time
#   saccade duration/amplitude within trial-time

# TODO:
#   micro-saccade rate relative to identification time
