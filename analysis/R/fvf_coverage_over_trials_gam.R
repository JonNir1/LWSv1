
#' === ARRAY COVERAGE OVER TRIAL NUMBER ===
#'
#' Does the fraction of icons within a subject's FVF (array coverage, from `array_coverage.ipynb`) shrink or
#' grow between trial 1 and trial 60? Companion to `fvf_over_trials_gam.R`, which models the FVF radius
#' itself (only possible for estimator D). This script instead asks a different question - does the
#' *coverage* achieved under a given FVF radius change over the session - and is fit once per FVF estimator
#' (C and D) by the caller sourcing this file twice, once per estimator's `dat`.
#'
#' Sourced via rpy2 (analysis/helpers/r_bridge.py::source_r), not run via Rscript - no CSV I/O here. The
#' caller builds `dat` (subject, trial, n_icons, n_covered) in R via r_bridge.py::to_r_dataframe() before
#' sourcing this file, and reads `model` back out afterward via r_bridge.py::get_r_object(). Trial-level
#' grain, same reasoning as `fvf_over_trials_gam.R` for why there is no flat/nested pair here.


library(mgcv)

K <- 10

# fit GAM with trial num as predictor; response is a per-trial proportion (icons covered / n_icons), modeled
# as a binomial count so the variance follows the actual trial size rather than assuming constant variance.
model <- gam(
  cbind(n_covered, n_icons - n_covered) ~ s(trial, k = K, bs = "tp") + s(subject, bs = "re"),
  data = dat,
  family = binomial(),
  method = "REML"
)
