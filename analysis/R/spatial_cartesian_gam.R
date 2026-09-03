
#' === LWS PROBABILITY OVER SCREEN POSITION (CARTESIAN) ===
#'
#' Does LWS probability vary with where on the screen the target sits, and does that pattern differ by
#' trial type? Sourced via rpy2 (analysis/helpers/r_bridge.py::source_r), not run via Rscript - no CSV I/O
#' here. The caller builds `dat` (subject, trial, trial_category, trial_uid, x, y, is_lws) in R via
#' r_bridge.py::to_r_dataframe() before sourcing this file, and reads `simple_model_flat`/
#' `simple_model_nested`/`interaction_model` back out afterward via r_bridge.py::get_r_object().
#' See analysis/R/spatial_polar_gam.R for the eccentricity/polar-coordinates counterpart.


library(mgcv)

K <- 8  # per *marginal* basis in te(), so the tensor holds ~K^2 basis functions

# M10 severity audit: simple_model_flat is the "ill-conceived" model CODE_REVIEW.md M10 originally flagged -
# s(subject, bs="re") only, even though the unit of observation is a visit, and visits nest within trial
# within subject. nthreads=1: discrete=TRUE's OpenMP threading segfaults this embedded-R-via-rpy2 session
# when fitting more than one model per session - see r_bridge.py.
simple_model_flat <- bam(
  is_lws ~ trial_category + te(x, y, k = K, bs = "tp") + s(subject, bs = "re"),
  data = dat,
  family = binomial(),
  method = "fREML",
  discrete = TRUE, nthreads = 1
)

# CODE_REVIEW.md M10, fixed: s(trial_uid, bs="re") absorbs the visit-within-trial-within-subject nesting
# alongside the subject-level random intercept, so the smooth's p-value is no longer anti-conservative.
simple_model_nested <- bam(
  is_lws ~ trial_category + te(x, y, k = K, bs = "tp") + s(subject, bs = "re") + s(trial_uid, bs = "re"),
  data = dat,
  family = binomial(),
  method = "fREML",
  discrete = TRUE, nthreads = 1
)

# Does the spatial pattern differ by trial type? Kept from the original M10/M11 fix session as-is (nested
# only - see CODE_REVIEW.md and this session's M10-audit report for why it doesn't get its own flat/nested
# pair: it's a secondary decomposition of the same te(x,y) main effect simple_model already tests, and a
# separate pair would very likely just re-demonstrate the same M10 effect a third time).
#
# CODE_REVIEW.md M11, fixed - but not the way first proposed: anova() needs likelihoods comparable across
# the two models' differing smooth structure, which fREML fits are not, and the textbook fix (refit under
# method="ML") isn't reachable here (discrete=TRUE only supports fREML/NCV; the non-discretized ML refit
# did not finish in 5+ hours of CPU time). Instead: mgcv's `by`-factor smooth already gives a valid
# within-fit test of "does trial type change the spatial pattern" - each level of te(x, y, by=trial_category)
# is fit and tested (edf, p-value) inside this model's own fREML fit, no second model or refit needed.
interaction_model <- bam(
  is_lws ~ trial_category +
    te(x, y, k = K, bs = "tp") +
    te(x, y, k = K, bs = "tp", by = trial_category) +
    s(subject, bs = "re") +
    s(trial_uid, bs = "re"),
  data = dat,
  family = binomial(),
  method = "fREML",
  discrete = TRUE, nthreads = 1
)
