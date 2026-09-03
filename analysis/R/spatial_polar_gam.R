
#' === LWS PROBABILITY BY ECCENTRICITY AND ANGLE (POLAR) ===
#'
#' te(x, y) in analysis/R/spatial_cartesian_gam.R can, in principle, fit an arbitrary 2-D surface, including
#' a radially symmetric one - so it isn't that Cartesian coordinates *can't* capture an eccentricity effect,
#' it's that te(x, y) doesn't *isolate* one from incidental left/right or top/bottom idiosyncrasies. s(r) +
#' s(theta) is a deliberately narrower, additive model (no r*theta interaction): "the effect, if any,
#' depends on distance from center and/or angle, each on its own." It spends its (much smaller) degrees of
#' freedom on that specific shape instead of sharing them with an unconstrained 2-D surface, so a genuine
#' eccentricity or meridian effect is easier to detect and interpret here than in te(x, y) - and a high-edf
#' result here would be as suspicious as a high-edf te(x, y).
#'
#' H0: no effect of distance-from-center (r) and no effect of angle (theta) on P[LWS]. Under H0, s(r) and
#' s(theta) should each shrink toward a flat (constant) function - edf near 0 - not toward a linear one. A
#' low-but-nonzero edf (roughly 1-3) is a simple, credible, interpretable effect; edf close to K is as
#' likely to reflect the model fitting noise as a real pattern.
#'
#' Sourced via rpy2 (analysis/helpers/r_bridge.py::source_r), not run via Rscript - no CSV I/O here. The
#' caller builds `dat` (subject, trial, trial_category, trial_uid, x, y, is_lws) in R via
#' r_bridge.py::to_r_dataframe() before sourcing this file, and reads `eccentricity_model_flat`/
#' `eccentricity_model_nested` back out afterward via r_bridge.py::get_r_object(). `r`/`theta` are computed
#' here from x/y (see analysis/R/spatial_cartesian_gam.R).


library(mgcv)

# CENTER_X/CENTER_Y must match constants.py's TOBII_MONITOR (1920x1080, top-left pixel origin) - kept in raw
# pixels, consistent with the Cartesian model and with CODE_REVIEW.md M12's caveat that px2deg's
# eccentricity bias would otherwise be baked into the very metric this section is testing.
CENTER_X <- 960
CENTER_Y <- 540
Kr <- 10   # basis size for s(r)
Kth <- 8   # basis size for s(theta); modest, since a meridian effect (if real) is expected to be a coarse pattern

dat$r <- sqrt((dat$x - CENTER_X)^2 + (dat$y - CENTER_Y)^2)                     # px, distance from screen center
dat$theta <- (atan2(-(dat$y - CENTER_Y), dat$x - CENTER_X) * 180 / pi) %% 360  # deg, 0=right, 90=up, wraps 0/360

# M10 severity audit: eccentricity_model_flat is the "ill-conceived" model - s(subject, bs="re") only, even
# though the unit of observation is a visit, and visits nest within trial within subject. nthreads=1:
# discrete=TRUE's OpenMP threading segfaults this embedded-R-via-rpy2 session when fitting more than one
# model per session - see r_bridge.py.
eccentricity_model_flat <- bam(
  is_lws ~ trial_category + s(r, k = Kr, bs = "tp") + s(theta, k = Kth, bs = "cc") + s(subject, bs = "re"),
  data = dat,
  family = binomial(),
  method = "fREML",
  discrete = TRUE, nthreads = 1,
  knots = list(theta = c(0, 360))  # cyclic spline needs an explicit period, or it wraps at data's min/max instead of 0/360
)

# CODE_REVIEW.md M10, fixed: s(trial_uid, bs="re") absorbs the visit-within-trial-within-subject nesting
# alongside the subject-level random intercept, so the smooths' p-values are no longer anti-conservative.
eccentricity_model_nested <- bam(
  is_lws ~ trial_category + s(r, k = Kr, bs = "tp") + s(theta, k = Kth, bs = "cc") +
    s(subject, bs = "re") + s(trial_uid, bs = "re"),
  data = dat,
  family = binomial(),
  method = "fREML",
  discrete = TRUE, nthreads = 1,
  knots = list(theta = c(0, 360))
)
