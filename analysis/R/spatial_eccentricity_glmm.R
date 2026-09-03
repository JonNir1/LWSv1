
#' === PARAMETRIC (LOG-LINEAR) ROBUSTNESS CHECK FOR THE ECCENTRICITY GAM ===
#'
#' spatial_gam.R's eccentricity model found s(r) non-linear (edf ~= 2.9). This script fits a genuine
#' parametric restriction of that smooth - ln(r) instead of s(r), sin(theta)/cos(theta) (a single-harmonic
#' Fourier term) instead of s(theta) - as a robustness check: does a principled, monotonic parametric shape
#' corroborate the GAM's finding, or was the GAM's significance an artefact of its flexibility?
#'
#' Originally meant to be fit via pymer4 directly in the notebook, but this session's environment can't
#' initialize rpy2 (R CMD config --ldflags needs Rtools/make, not installed here) - the same failure
#' spatial_effects.ipynb's own abandoned rpy2 cell already hit. Fit here via plain lme4::glmer() instead,
#' matching this directory's existing *_gam.R architecture, which never touches rpy2 at all.


.libPaths(c("C:/Users/nirjo/R_library/4.5", .libPaths()))  # lme4 lives in the user library, not the system one
library(lme4)
source(file.path("analysis", "R", "helpers.R"))

CENTER_X <- 960
CENTER_Y <- 540

dat <- load_data(
  file.path("analysis", "R", "funnel_results.csv"),
  valid_only = TRUE,
  on_target_only = TRUE
)
dat$r <- sqrt((dat$x - CENTER_X)^2 + (dat$y - CENTER_Y)^2)
theta_rad <- atan2(-(dat$y - CENTER_Y), dat$x - CENTER_X)
dat$log_r <- log(dat$r)
dat$sin_theta <- sin(theta_rad)
dat$cos_theta <- cos(theta_rad)

# (1 | subject/trial): lme4's native nested-random-effect shorthand, equivalent to the trial_uid workaround
# spatial_gam.R needs for mgcv - lme4 builds the subject:trial interaction internally.
model <- glmer(
  is_lws ~ trial_category + log_r + sin_theta + cos_theta + (1 | subject/trial),
  data = dat,
  family = binomial(),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

print(summary(model))
cat("\nconvergence check:", is.null(model@optinfo$conv$lme4$messages), "(TRUE = no convergence warnings)\n")

coefs <- as.data.frame(summary(model)$coefficients)
coefs$term <- rownames(coefs)
outfile <- file.path("analysis", "R", "spatial_eccentricity_glmm_coefs.csv")
write.csv(coefs, outfile, row.names = FALSE)
message("coefficients written to ", outfile)
