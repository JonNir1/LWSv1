#' === SAME-ICON VS. DIFFERENT-ICON EFFECT ON LWS PROBABILITY (Q5) ===
#'
#' Sourced via rpy2 (analysis/helpers/r_bridge.py::source_r), not run via Rscript - no CSV I/O here. The
#' caller builds `dat` (subject, trial, trial_category, same_icon_as_last_hit, is_lws), restricted to
#' `num_targets_found_before > 0` (the predictor is undefined otherwise), via r_bridge.py::to_r_dataframe()
#' before sourcing this file. `trial` is a real column, so lme4's native `(1 | subject/trial)` shorthand
#' needs no `trial_uid` workaround.

library(lme4)

GLMER_CONTROL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

model_flat <- glmer(
  is_lws ~ same_icon_as_last_hit + trial_category + (1 | subject),
  data = dat, family = binomial(), control = GLMER_CONTROL
)
model_nested <- glmer(
  is_lws ~ same_icon_as_last_hit + trial_category + (1 | subject/trial),
  data = dat, family = binomial(), control = GLMER_CONTROL
)
