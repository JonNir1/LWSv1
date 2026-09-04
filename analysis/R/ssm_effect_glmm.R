#' === SSM/SoS EFFECT ON LWS PROBABILITY (Q1, Q3, Q4) ===
#'
#' Sourced via rpy2 (analysis/helpers/r_bridge.py::source_r), not run via Rscript - no CSV I/O here. The
#' caller builds `dat` (subject, trial, trial_category, target_category, any_prior_hit, is_lws) in R via
#' r_bridge.py::to_r_dataframe() before sourcing this file, and reads the six model objects back out
#' afterward via r_bridge.py::get_r_object(). `trial` is a real column in `dat`, so lme4's native
#' `(1 | subject/trial)` shorthand needs no `trial_uid` workaround (unlike the mgcv GAM scripts).
#'
#' Every formula gets a flat `(1|subject)` fit (the CODE_REVIEW.md M10 "ill-conceived" baseline) and a
#' `(1|subject/trial)` nested companion, since the unit of observation is a visit and visits nest within
#' trial within subject exactly as in the other funnel-based analyses.


library(lme4)

GLMER_CONTROL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

# Q1: is there an SSM/SoS effect at all? Headline test - the `any_prior_hitTRUE` coefficient.
m1_flat <- glmer(
  is_lws ~ any_prior_hit + (1 | subject),
  data = dat, family = binomial(), control = GLMER_CONTROL
)
m1_nested <- glmer(
  is_lws ~ any_prior_hit + (1 | subject/trial),
  data = dat, family = binomial(), control = GLMER_CONTROL
)

# Q3: does the effect differ by trial type (COLOR/BW/NOISE)?
m3_flat <- glmer(
  is_lws ~ any_prior_hit * trial_category + (1 | subject),
  data = dat, family = binomial(), control = GLMER_CONTROL
)
m3_nested <- glmer(
  is_lws ~ any_prior_hit * trial_category + (1 | subject/trial),
  data = dat, family = binomial(), control = GLMER_CONTROL
)

# Q4: does the effect differ by the *missed* target's category (6-level ImageCategoryEnum)?
m4_flat <- glmer(
  is_lws ~ any_prior_hit * target_category + trial_category + (1 | subject),
  data = dat, family = binomial(), control = GLMER_CONTROL
)
m4_nested <- glmer(
  is_lws ~ any_prior_hit * target_category + trial_category + (1 | subject/trial),
  data = dat, family = binomial(), control = GLMER_CONTROL
)
