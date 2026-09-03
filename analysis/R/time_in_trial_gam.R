
#' === LWS PROBABILITY OVER TIME IN TRIAL ===
#'
#' Sourced via rpy2 (analysis/helpers/r_bridge.py::source_r), not run via Rscript - no CSV I/O here. The
#' caller builds `dat` (subject, trial, trial_category, trial_uid, start_time, is_lws) in R via
#' r_bridge.py::to_r_dataframe() before sourcing this file, and reads `model_flat`/`model_nested` back out
#' afterward via r_bridge.py::get_r_object().


library(mgcv)

K <- 10

# M10 severity audit: model_flat is the "ill-conceived" model CODE_REVIEW.md M10 originally flagged -
# s(subject, bs="re") only, even though the unit of observation is a visit, and visits nest within trial
# within subject.
model_flat <- bam(
  is_lws ~ trial_category + s(start_time, k = K, bs = "tp") + s(subject, bs = "re"),
  data = dat,
  family = binomial(),
  method = "fREML",
  discrete = TRUE, nthreads = 1  # discrete=TRUE's OpenMP threading segfaults when this R session is embedded via rpy2 - see r_bridge.py
)

# CODE_REVIEW.md M10, fixed: s(trial_uid, bs="re") (trial_uid = subject:trial, built by
# to_r_dataframe()) absorbs the visit-within-trial-within-subject nesting alongside the subject-level
# random intercept, so the smooth's p-value is no longer anti-conservative.
model_nested <- bam(
  is_lws ~ trial_category + s(start_time, k = K, bs = "tp") + s(subject, bs = "re") + s(trial_uid, bs = "re"),
  data = dat,
  family = binomial(),
  method = "fREML",
  discrete = TRUE, nthreads = 1  # discrete=TRUE's OpenMP threading segfaults when this R session is embedded via rpy2 - see r_bridge.py
)
