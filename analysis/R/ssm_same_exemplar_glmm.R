#' === SAME-EXEMPLAR-DIFFERENT-LOCATION EFFECT ON LWS PROBABILITY (Q5) ===
#'
#' Sourced via rpy2 (analysis/helpers/r_bridge.py::source_r), not run via Rscript - no CSV I/O here. The
#' caller builds `dat` (subject, trial, trial_category, same_exemplar_as_last_hit, is_lws), restricted to
#' `num_targets_found_before > 0` AND `same_icon_as_last_hit == FALSE` (a target's exact image/`sub_path`
#' can appear at up to two array positions in a trial; a visit to the *literal same position* as the most
#' recent hit is a `VisitType.TARGET_RETURN` and structurally never LWS, so it's excluded rather than
#' analyzed as if it were a "same" case - see `ssm.py::add_ssm_predictors`), via
#' r_bridge.py::to_r_dataframe() before sourcing this file. `trial` is a real column, so lme4's native
#' `(1 | subject/trial)` shorthand needs no `trial_uid` workaround.

library(lme4)

GLMER_CONTROL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

model_flat <- glmer(
  is_lws ~ same_exemplar_as_last_hit + trial_category + (1 | subject),
  data = dat, family = binomial(), control = GLMER_CONTROL
)
model_nested <- glmer(
  is_lws ~ same_exemplar_as_last_hit + trial_category + (1 | subject/trial),
  data = dat, family = binomial(), control = GLMER_CONTROL
)
