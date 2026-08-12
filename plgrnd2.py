import time

import bambi as bmb

import plotly.io as pio

import config as cnfg

pio.renderers.default = "browser"


# %%
# ##  Run Pipeline
# from pipeline.stage1_parse.run_stage1 import run_stage1
# targets, actions, metadata, idents, fixations, visits = run_pipeline(
#     # raw_data_path=cnfg.RAW_DATA_PATH,
#     # identification_actions=cnfg.IDENTIFICATION_ACTIONS,
#     # on_target_threshold_dva=cnfg.ON_TARGET_THRESHOLD_DVA,
#     # gaze_to_trigger_time_threshold=cnfg.MAX_GAZE_TO_TRIGGER_TIME_DIFF,
#     # visit_merging_time_threshold=cnfg.VISIT_MERGING_TIME_THRESHOLD,
#     save=True,
#     verbose=True
# )


# %%
# ##  Load Data

from analysis.helpers.read_data import load_data

data = load_data(cnfg.OUTPUT_PATH, drop_bad_eye=True, drop_outliers=True)
targets = data.targets
actions = data.actions
metadata = data.metadata
idents = data.identifications
fixations = data.fixations
visits = data.visits


# %%
from pipeline.stage3_classify.build_funnels import build_trial_inclusion_funnel, build_event_classification_funnel
from analysis.helpers.visualizations.funnel.size_and_proportion import calculate_step_sizes
from pipeline.config import (
    TRIAL_INCLUSION_CRITERIA, IS_LWS_CRITERIA, IS_TARGET_RETURN_CRITERIA, cumulative_names,
)

trial_funnel = build_trial_inclusion_funnel(data)
trial_funnel_sizes = calculate_step_sizes(
    trial_funnel,
    ["subject", "trial"],
    cumulative_names(TRIAL_INCLUSION_CRITERIA + ["is_valid_trial"])
)

is_lws_funnel = build_event_classification_funnel(
    data,
    event_type="visit",
    funnel_type="lws",
)
lws_sizes = calculate_step_sizes(
    is_lws_funnel,
    ["subject", "trial", "trial_category", "target_category"],
    cumulative_names(IS_LWS_CRITERIA + ["is_lws"])
)

is_tr_funnel = build_event_classification_funnel(
    data,
    event_type="visit",
    funnel_type="target_return",
)
tr_funnel_sizes = calculate_step_sizes(
    is_tr_funnel,
    ["subject", "trial", "trial_category", "target_category"],
    cumulative_names(IS_TARGET_RETURN_CRITERIA + ["is_target_return"])
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
