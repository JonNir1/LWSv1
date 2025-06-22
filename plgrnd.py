import numpy as np
import pandas as pd
import plotly.io as pio

import config as cnfg
from data_models.LWSEnums import DominantEyeEnum
from data_models.Subject import Subject

pio.renderers.default = "browser"


# read subject data
# subj = Subject.from_raw(
#     exp_name=cnfg.EXPERIMENT_NAME, subject_id=1, session=1, data_dir="v4-1-1 GalChen Demo", verbose=True
# )
# subj.to_pickle(overwrite=False)
subj = Subject.from_pickle(exp_name=cnfg.EXPERIMENT_NAME, subject_id=1,)

## process subject data
identifications_df = subj.get_target_identification_summary(verbose=False)

from analysis.fixations import get_fixations
fixations_df = get_fixations(subj, save=False, verbose=True)
fixations_df = fixations_df[fixations_df["outlier_reasons"].apply(lambda x: len(x) == 0)]   # drop outliers

from analysis.visits import get_visits
visits_df = get_visits(subj, save=True, verbose=True)



## LWS FUNNEL ##
metadata = subj.get_metadata(verbose=False)
valid_trials = list(metadata[~metadata[f"bad_{cnfg.TRIAL_STR}"].astype(bool)].index)

# convert fixations_df to lws-funnel format
fixations = fixations_df.rename(
    columns={f"closest_{cnfg.TARGET_STR}": cnfg.TARGET_STR}
).set_index(cnfg.TARGET_STR, append=True).reorder_levels(
    [cnfg.TRIAL_STR, cnfg.TARGET_STR, cnfg.EYE_STR, cnfg.FIXATION_STR]
).sort_index()
# keep the distance to the only relevant target
fixations[cnfg.TARGET_DISTANCE_STR] = pd.Series(fixations.to_numpy()[
    np.arange(len(fixations)), fixations.columns.get_indexer(fixations.index.get_level_values(cnfg.TARGET_STR))
], index=fixations.index)
fixations.drop(columns=[
    col for col in fixations.columns if col.startswith(cnfg.TARGET_STR) and col != cnfg.TARGET_DISTANCE_STR
], inplace=True)
# add target detection columns
fixations = fixations.reset_index(drop=False).set_index([cnfg.TRIAL_STR, cnfg.TARGET_STR])
idents = identifications_df.set_index([cnfg.TRIAL_STR, cnfg.TARGET_STR]).rename(
    columns={cnfg.TIME_STR: cnfg.TARGET_TIME_STR}
).loc[fixations.index]
fixations = pd.concat([fixations, idents], axis=1).reset_index(drop=False)


# calculate the funnel
metadata = subj.get_metadata(verbose=False)
valid_trials = list(metadata[~metadata[f"bad_{cnfg.TRIAL_STR}"].astype(bool)].index)

lws_funnel_dict = dict()
lws_funnel_dict['all'] = fixations
lws_funnel_dict['valid_trials'] = lws_funnel_dict['all'][
    ~(lws_funnel_dict['all'][f"bad_{cnfg.TRIAL_STR}"].astype(bool))
]
lws_funnel_dict['not_outlier'] = lws_funnel_dict['valid_trials'][
    lws_funnel_dict['valid_trials']["outlier_reasons"].map(lambda val: len(val) == 0)
]
lws_funnel_dict['on_target'] = lws_funnel_dict['not_outlier'][
    lws_funnel_dict['not_outlier'][cnfg.TARGET_DISTANCE_STR] <= (cnfg.ON_TARGET_THRESHOLD_DVA / subj.px2deg)
]
lws_funnel_dict['before_identification'] = lws_funnel_dict['on_target'][
    lws_funnel_dict['on_target'][f"{cnfg.END_TIME_STR}"] <= lws_funnel_dict['on_target'][cnfg.TARGET_TIME_STR]
]
lws_funnel_dict['next_1_not_in_strip'] = lws_funnel_dict['before_identification'][
    ~(lws_funnel_dict['before_identification']['next_1_in_strip'].astype(bool))
]
lws_funnel_dict['next_2_not_in_strip'] = lws_funnel_dict['next_1_not_in_strip'][
    ~(lws_funnel_dict['next_1_not_in_strip']['next_1_in_strip'].astype(bool)) &
    ~(lws_funnel_dict['next_1_not_in_strip']['next_2_in_strip'].astype(bool))
]
lws_funnel_dict['next_3_not_in_strip'] = lws_funnel_dict['next_2_not_in_strip'][
    ~(lws_funnel_dict['next_2_not_in_strip']['next_1_in_strip'].astype(bool)) &
    ~(lws_funnel_dict['next_2_not_in_strip']['next_2_in_strip'].astype(bool)) &
    ~(lws_funnel_dict['next_2_not_in_strip']['next_3_in_strip'].astype(bool))
]
lws_funnel_dict['not_end_with_trial'] = lws_funnel_dict['next_3_not_in_strip'][

]


def lws_funnel(events: pd.DataFrame, trial_metadata: pd.DataFrame):
    return None



# ### Single-Subject Identification Figures ###
#
# # identification (space-key hit) rate, time, distance
# from analysis.within_subject_figures.identification_figures import identification_rate_figure
# fig = identification_rate_figure(identifications_df)
# fig.show()
#
# from analysis.within_subject_figures.identification_figures import identification_time_figure
# fig = identification_time_figure(identifications_df)
# fig.show()
#
# from analysis.within_subject_figures.identification_figures import identification_distance_figure
# fig = identification_distance_figure(identifications_df, subj.px2deg)
# fig.show()
#
# # identification-event start-time
# from analysis.within_subject_figures.identification_figures import identification_event_start_time_figure
# fig = identification_event_start_time_figure(identifications_df, fixations_df, cnfg.FIXATION_STR, dominant_eye=subj.eye)
# fig.show()
#
# fig = identification_event_start_time_figure(identifications_df, visits_df, cnfg.VISIT_STR, dominant_eye=subj.eye)
# fig.show()
#
#
# # identification-event distance from targets
# from analysis.within_subject_figures.identification_figures import identification_event_distance_figure
# fig = identification_event_distance_figure(identifications_df, fixations_df, cnfg.FIXATION_STR, subj.px2deg, dominant_eye=subj.eye)
# fig.show()
#
# fig = identification_event_distance_figure(identifications_df, visits_df, cnfg.VISIT_STR, subj.px2deg, dominant_eye=subj.eye)
# fig.show()














### LWS IDENTIFICATION ###

# check if fixations don't end too close to the trial end
MIN_TIME_FROM_TRIAL_END = 100   # ms
is_trial_end = fixations_df["to_trial_end"] <= MIN_TIME_FROM_TRIAL_END
del MIN_TIME_FROM_TRIAL_END


# check if **current or 1-next** fixations end before target marked
is_before_marking = pd.DataFrame(
    fixations_df["end_time"].values < targets_df["time"].values[:, np.newaxis],
    columns=fixations_df.index, index=targets_df.index,
).T
is_next_before_marking = pd.concat([
    is_before_marking.xs(DominantEyeEnum.LEFT, level=cnfg.EYE_STR).shift(-1),
    is_before_marking.xs(DominantEyeEnum.RIGHT, level=cnfg.EYE_STR).shift(-1)
], keys=[cnfg.LEFT_STR, cnfg.RIGHT_STR], axis=0)
is_next_before_marking.index = is_next_before_marking.index.reorder_levels([1, 0, 2]).rename(is_before_marking.index.names)
# fill nans due to shifting:
is_next_before_marking = is_next_before_marking.mask(is_next_before_marking.isna(), is_before_marking)


# check if the **current or 1-next** fixation is in the bottom strip
is_in_strip = fixs_df['in_strip']
is_next_in_strip = pd.concat([
    is_in_strip.xs(DominantEyeEnum.LEFT, level=cnfg.EYE_STR).shift(-1),
    is_in_strip.xs(DominantEyeEnum.RIGHT, level=cnfg.EYE_STR).shift(-1)
], keys=[cnfg.LEFT_STR, cnfg.RIGHT_STR], axis=0)
is_next_in_strip.index = is_next_in_strip.index.reorder_levels([1, 0, 2]).rename(is_in_strip.index.names)
# fill nans due to shifting:
is_next_in_strip = is_next_in_strip.mask(is_next_in_strip.isna(), is_in_strip)


# number fixations that are close to each target, relative to target-marking (on marking = 0)
MAX_DEG_FROM_TARGET = 1.0
px_distances = fixations_df[[col for col in fixations_df.columns if col.startswith(cnfg.TARGET_STR)]]
is_on_target = px_distances <= (MAX_DEG_FROM_TARGET / subj.px2deg)
num_on_target = pd.concat([
    is_on_target.xs(DominantEyeEnum.LEFT, level=cnfg.EYE_STR).cumsum(),
    is_on_target.xs(DominantEyeEnum.RIGHT, level=cnfg.EYE_STR).cumsum()
], keys=[cnfg.LEFT_STR, cnfg.RIGHT_STR], axis=0).astype(float)
num_on_target.index = num_on_target.index.reorder_levels([1, 0, 2]).rename(is_on_target.index.names)
num_on_target = num_on_target.where(is_on_target)

marking_fix_num = num_on_target.loc[fixations_df['curr_marked'].notnull()]   # num-on-target for the fixation where target was marked
marking_fix_num = marking_fix_num.stack().droplevel('fixation').unstack(cnfg.EYE_STR).T
num_from_mark = num_on_target - marking_fix_num
del MAX_DEG_FROM_TARGET, px_distances, is_on_target, num_on_target, marking_fix_num




# LWS instance:
#   not coinciding with trial end
#   fixation on target
#   before identification
#   succeeding fixation is not in bottom strip
#   succeeding fixation is not same-target's identification fixation
# TODO: determine if fixation close to target, number proximal fixations
