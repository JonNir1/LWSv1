import time

import bambi as bmb

import plotly.io as pio

import config as cnfg

pio.renderers.default = "browser"


# %%
# ##  Run Pipeline
# from pipeline.run_pipeline import run_pipeline
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

from analysis.helpers.read_data import read_data

loaded_data = read_data(cnfg.OUTPUT_PATH, drop_bad_eye=True, drop_outliers=True)
targets = loaded_data.targets
actions = loaded_data.actions
metadata = loaded_data.metadata
idents = loaded_data.identifications
fixations = loaded_data.fixations
visits = loaded_data.visits
del loaded_data    # free up memory by deleting the loaded_data object


# %%
from analysis.helpers.funnels import build_trial_inclusion_funnel, build_event_classification_funnel, calculate_funnel_step_sizes
from analysis.helpers.funnels.funnel_config import TRIAL_INCLUSION_CRITERIA, IS_LWS_CRITERIA, IS_TARGET_RETURN_CRITERIA

trial_funnel = build_trial_inclusion_funnel(
    cnfg.OUTPUT_PATH
)
trial_funnel_sizes = calculate_funnel_step_sizes(
    trial_funnel,
    ["subject", "trial"],
    TRIAL_INCLUSION_CRITERIA + ["is_valid_trial"]
)

is_lws_funnel = build_event_classification_funnel(
    cnfg.OUTPUT_PATH,
    event_type="visit",
    funnel_type="lws",
)
lws_sizes = calculate_funnel_step_sizes(
    is_lws_funnel,
    ["subject", "trial", "trial_category", "target_category"],
    IS_LWS_CRITERIA + ["is_lws"]
)

is_tr_funnel = build_event_classification_funnel(
    cnfg.OUTPUT_PATH,
    event_type="visit",
    funnel_type="target_return",
)
tr_funnel_sizes = calculate_funnel_step_sizes(
    is_tr_funnel,
    ["subject", "trial", "trial_category", "target_category"],
    IS_TARGET_RETURN_CRITERIA + ["is_target_return"]
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
