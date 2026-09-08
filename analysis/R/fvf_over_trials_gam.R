
#' === FVF RADIUS OVER TRIAL NUMBER ===
#'
#' Does the functional visual field radius itself drift between trial 1 and trial 60? The response here is a
#' per-trial FVF estimate (DVA), not a derived coverage count - only estimator D (encircling criterion,
#' `analysis/fvf/fvf.py::estimate_by_encircling`) produces one: it is computed per (subject, trial) directly.
#' Estimator C (selection hazard) needs many pooled opportunities to trace its hazard curve and only ever
#' returns one value per *subject*, so it cannot supply this table - this script is D-only.
#'
#' Sourced via rpy2 (analysis/helpers/r_bridge.py::source_r), not run via Rscript - no CSV I/O here. The
#' caller builds `dat` (subject, trial, fvf_dva) in R via r_bridge.py::to_r_dataframe() before sourcing this
#' file, and reads `model` back out afterward via r_bridge.py::get_r_object(). Trial-level grain (one row per
#' subject/trial, no repeated within-trial observations), so no flat/nested random-effect pair - out of
#' scope for the M10 severity audit for the same reason as CODE_REVIEW.md already documents.


library(mgcv)

K <- 10

# fit GAM with trial num as predictor; response is the per-trial FVF radius (DVA) directly, so a plain
# Gaussian GAM - no count/proportion structure to model here, unlike the array-coverage response this
# replaced.
model <- gam(
  fvf_dva ~ s(trial, k = K, bs = "tp") + s(subject, bs = "re"),
  data = dat,
  method = "REML"
)
