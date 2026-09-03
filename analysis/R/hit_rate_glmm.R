
#' === M10 SEVERITY AUDIT: HIT RATE AGAINST STIMULUS FEATURES ===
#'
#' hit_rate.ipynb's existing model (`is_hit ~ abs_target_angle * trial_category * target_category +
#' (1 | subject)`, fit via pymer4/glmer) never converged (see the notebook's own saved output:
#' `Convergence status: [1] FALSE`). Refit here via plain lme4::glmer(), for a clean, directly-comparable
#' coefficient/AIC/BIC/R^2 table (the original pymer4 fit isn't easily re-extracted the same way), also
#' switching `abs_target_angle` -> `centered_abs_target_angle` since we're refitting anyway and every other
#' analysis in this repo already centers it (mean-centering interaction terms reduces collinearity between
#' main effects and interactions, which may also help the convergence problem).
#'
#' Sourced via rpy2 (analysis/helpers/r_bridge.py::source_r), not run via Rscript - no CSV I/O here. The
#' caller builds `dat` (subject, trial, trial_category, target_category, centered_abs_target_angle, is_hit)
#' in R via r_bridge.py::to_r_dataframe() before sourcing this file, and reads `model_flat`/`model_nested`
#' back out afterward via r_bridge.py::get_r_object(). `trial` is already a real column in `dat` (unlike the
#' funnel-derived analyses), so lme4's native `(1 | subject/trial)` shorthand needs no `trial_uid` workaround.


library(lme4)

GLMER_CONTROL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

# M10 severity audit: model_flat is the "ill-conceived" model - (1|subject) only, even though the unit of
# observation (an identification action) nests within trial within subject.
model_flat <- glmer(
  is_hit ~ centered_abs_target_angle * trial_category * target_category + (1 | subject),
  data = dat,
  family = binomial(),
  control = GLMER_CONTROL
)

# CODE_REVIEW.md M10, fixed via lme4's native nested-random-effect shorthand: (1 | subject/trial) builds
# the subject:trial interaction internally, absorbing the identification-within-trial-within-subject nesting.
model_nested <- glmer(
  is_hit ~ centered_abs_target_angle * trial_category * target_category + (1 | subject/trial),
  data = dat,
  family = binomial(),
  control = GLMER_CONTROL
)
