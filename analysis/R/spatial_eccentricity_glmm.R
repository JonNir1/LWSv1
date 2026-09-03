
#' === PARAMETRIC (LOG-LINEAR) ROBUSTNESS CHECK FOR THE ECCENTRICITY GAM ===
#'
#' spatial_polar_gam.R's eccentricity model found s(r) non-linear (edf ~= 2.9). This script fits a genuine
#' parametric restriction of that smooth - ln(r) instead of s(r), sin(theta)/cos(theta) (a single-harmonic
#' Fourier term) instead of s(theta) - as a robustness check: does a principled, monotonic parametric shape
#' corroborate the GAM's finding, or was the GAM's significance an artefact of its flexibility?
#'
#' Sourced via rpy2 (analysis/helpers/r_bridge.py::source_r), not run via Rscript - no CSV I/O here (fit
#' via plain lme4::glmer(), matching this directory's *_gam.R architecture but with no spline terms, so no
#' `discrete=TRUE`/`nthreads` concern here - lme4 has no such threading issue). The caller builds `dat`
#' (subject, trial, trial_category, trial_uid, x, y, is_lws) in R via r_bridge.py::to_r_dataframe() before
#' sourcing this file, and reads `model_flat`/`model_nested` back out afterward via
#' r_bridge.py::get_r_object().


library(lme4)

CENTER_X <- 960
CENTER_Y <- 540

dat$r <- sqrt((dat$x - CENTER_X)^2 + (dat$y - CENTER_Y)^2)
theta_rad <- atan2(-(dat$y - CENTER_Y), dat$x - CENTER_X)
dat$log_r <- log(dat$r)
dat$sin_theta <- sin(theta_rad)
dat$cos_theta <- cos(theta_rad)

GLMER_CONTROL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

# M10 severity audit: model_flat is the "ill-conceived" model - (1|subject) only, even though the unit of
# observation is a visit, and visits nest within trial within subject.
model_flat <- glmer(
  is_lws ~ trial_category + log_r + sin_theta + cos_theta + (1 | subject),
  data = dat,
  family = binomial(),
  control = GLMER_CONTROL
)

# CODE_REVIEW.md M10, fixed via lme4's native nested-random-effect shorthand: (1 | subject/trial) builds
# the subject:trial interaction internally, absorbing the visit-within-trial-within-subject nesting.
model_nested <- glmer(
  is_lws ~ trial_category + log_r + sin_theta + cos_theta + (1 | subject/trial),
  data = dat,
  family = binomial(),
  control = GLMER_CONTROL
)
