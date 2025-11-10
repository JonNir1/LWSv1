import time

import numpy as np
import pandas as pd
import bambi as bmb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.io as pio

import config as cnfg

pio.renderers.default = "browser"

# TODO: only from HOME:
cnfg.OUTPUT_PATH = r'C:\Users\nirjo\Desktop\LWS\Results'


# TODO (Nov 06): write notebook with
#   funnel visualization - total event count -> valid & on-target count -> LWS count
#   BAMBI model fitting for LWS (or Target-Return) funnel proportions
#       - use `valid & on-target` as initial step
#       - include trial_category, target_category and interaction as predictors (fixed effects)
#       - include subject as random effect (intercept + slopes)
#       - check if model converges for per-subject trial_category/target_category slopes (random effects)
#         if not, try only random intercepts
#       - check model convergence (`ess_bulk`, `r_hat`) and visualize trace plots of MCMC chains
#       - visualize posterior distributions of fixed effects + intercept
#       - conclude if there are main effects of trial_category/target_category and if there is interaction effect
#   Repeat analysis using a frequentist approach with R-lme4 - to compare results & conclusions


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

initial_step = "instance_on_target"

funnel_data = prepare_funnel(
    data_dir=cnfg.OUTPUT_PATH,
    funnel_type="lws",
    event_type="visit",
    verbose=True,
)


# %%
from analysis.funnel.prepare import get_funnel_steps
from analysis.funnel.proportion import calculate_funnel_sizes, calculate_proportions

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
# Fit `bambi` model
import arviz as az

# take only the subset of events where preliminary steps were passed (e.g. trial validity, outlier check, etc.)
funnel_subset = funnel_data[funnel_data[initial_step]]

# build the model:

start = time.time()

model_formula = "final ~ trial_category * target_category + (1 + trial_category + target_category | subject)"
model = bmb.Model(model_formula, funnel_subset, family="bernoulli")
idata = model.fit(
    draws=2000, tune=1000, chains=4, cores=2, target_accept=0.9, progressbar=True
)

z = az.summary(idata, var_names=["Intercept", "trial_category", "target_category"])

elapsed = time.time() - start
print(f"Model fitting completed in {elapsed // 3600}:{(elapsed % 3600) // 60}:{elapsed % 60:.2f} (hh:mm:ss)")

## model diagnostics
axes = az.plot_trace(idata, var_names=["Intercept", "trial_category", "target_category"])
fig = axes.ravel()[0].figure
fig.show()

## model posterior visualization
axes = az.plot_posterior(
    idata,
    var_names=["Intercept", "trial_category", "target_category"],
)
fig = axes.ravel()[0].figure
fig.show()


# %%
# deeper look into false alarms - what was the FAed target?

from helpers.sdt import calc_sdt_class_per_trial

has_hit_fa = (
    metadata
    .assign(
        has_hit=calc_sdt_class_per_trial(metadata, idents, "hit")["count"] > 0,
        has_fa=calc_sdt_class_per_trial(metadata, idents, "false_alarm")["count"] > 0,
    )
)
trials_hit_fa = has_hit_fa.loc[has_hit_fa["has_hit"] & has_hit_fa["has_fa"], ["subject", "trial", "trial_category"]]
idents_hit_fa = (
    idents
    .merge(
        trials_hit_fa,
        on=["subject", "trial"],
        how="inner",
    )
    .merge(
        targets[["subject", "trial", "target", "category"]],
        on=["subject", "trial", "target"],
        how="left",
    )
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
