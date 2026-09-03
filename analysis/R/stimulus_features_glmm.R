
#' === M10 SEVERITY AUDIT: LWS AGAINST STIMULUS FEATURES ===
#'
#' stimulus_features.ipynb's existing `freq_model` (`is_lws ~ trial_category * target_category *
#' centered_abs_target_angle + (1 | subject)`, fit via pymer4/glmer) converged, but only with a non-default
#' optimizer, and its own data-prep cell was stale/unexecuted in the saved notebook - not a trustworthy flat
#' baseline to pair against a new nested model. Refit both here via plain lme4::glmer(), fresh from
#' `funnel_results`, for a clean, directly-comparable coefficient/AIC/BIC/R^2 table.
#'
#' Sourced via rpy2 (analysis/helpers/r_bridge.py::source_r), not run via Rscript - no CSV I/O here. The
#' caller builds `dat` (subject, trial, trial_category, target_category, centered_abs_target_angle, is_lws)
#' in R via r_bridge.py::to_r_dataframe() before sourcing this file, and reads `model_flat`/`model_nested`
#' back out afterward via r_bridge.py::get_r_object(). `trial` is a real column here (not a funnel_uid
#' workaround), so lme4's native `(1 | subject/trial)` shorthand needs no extra construction.


library(lme4)

GLMER_CONTROL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

# M10 severity audit: model_flat is the "ill-conceived" model - (1|subject) only, even though the unit of
# observation is a visit, and visits nest within trial within subject.
model_flat <- glmer(
  is_lws ~ trial_category * target_category * centered_abs_target_angle + (1 | subject),
  data = dat,
  family = binomial(),
  control = GLMER_CONTROL
)

# CODE_REVIEW.md M10, fixed via lme4's native nested-random-effect shorthand: (1 | subject/trial) builds
# the subject:trial interaction internally, absorbing the visit-within-trial-within-subject nesting.
model_nested <- glmer(
  is_lws ~ trial_category * target_category * centered_abs_target_angle + (1 | subject/trial),
  data = dat,
  family = binomial(),
  control = GLMER_CONTROL
)
