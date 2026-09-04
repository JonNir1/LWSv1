#' === LWS PROBABILITY OVER TIME/FIXATIONS SINCE THE MOST RECENT HIT (Q2, Adamo et al. 2013 replication) ===
#'
#' Sourced via rpy2 (analysis/helpers/r_bridge.py::source_r), not run via Rscript - no CSV I/O here. The
#' caller builds `dat` (subject, trial, trial_uid, trial_category, time_since_recent_find,
#' fixations_since_last_hit, is_lws) via r_bridge.py::to_r_dataframe(build_trial_uid = TRUE) before sourcing
#' this file - `trial_uid` (subject:trial) is needed because, unlike lme4's `(1 | subject/trial)`, mgcv's
#' `s(x, bs = "re")` has no nested-random-effect shorthand (CODE_REVIEW.md M10).
#'
#' Two parallel smooths, fit and read back independently, since they test different accounts of the effect:
#' raw elapsed time (the literal attentional-blink lag) vs. number of intervening fixations (a fixation-lag
#' alternative robust to time/fixation-rate confounds). `dat` is already restricted to
#' `num_targets_found_before > 0` by the caller (the predictor is undefined otherwise).

library(mgcv)

K <- 10

# --- time_since_recent_find (ms) ---
model_time_flat <- bam(
  is_lws ~ trial_category + s(time_since_recent_find, k = K, bs = "tp") + s(subject, bs = "re"),
  data = dat, family = binomial(), method = "fREML",
  discrete = TRUE, nthreads = 1  # discrete=TRUE's OpenMP threading segfaults when embedded via rpy2 - see r_bridge.py
)
model_time_nested <- bam(
  is_lws ~ trial_category + s(time_since_recent_find, k = K, bs = "tp") + s(subject, bs = "re") + s(trial_uid, bs = "re"),
  data = dat, family = binomial(), method = "fREML",
  discrete = TRUE, nthreads = 1
)

# --- fixations_since_last_hit (count) ---
model_fix_flat <- bam(
  is_lws ~ trial_category + s(fixations_since_last_hit, k = K, bs = "tp") + s(subject, bs = "re"),
  data = dat, family = binomial(), method = "fREML",
  discrete = TRUE, nthreads = 1
)
model_fix_nested <- bam(
  is_lws ~ trial_category + s(fixations_since_last_hit, k = K, bs = "tp") + s(subject, bs = "re") + s(trial_uid, bs = "re"),
  data = dat, family = binomial(), method = "fREML",
  discrete = TRUE, nthreads = 1
)
