import plotly.io as pio

import config as cnfg

pio.renderers.default = "browser"

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
# ## Calculate & Show Funnels
from funnel.lws_funnel import lws_funnel
from funnel.helpers import calculate_funnel_sizes
from funnel.visualize import create_funnel_figure

funnel_results = lws_funnel(
    event_data=visits,
    metadata=metadata,
    actions=actions,
    idents=idents,
    event_type="visit",
)

funnel_sizes = calculate_funnel_sizes(funnel_results)
create_funnel_figure(
    funnel_sizes[funnel_sizes["eye"] == "right"], "lws", "visits", show_individuals=False
).show()


# %%

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
