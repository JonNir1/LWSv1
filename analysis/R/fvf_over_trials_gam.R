
#' === FVF RADIUS OVER TRIAL NUMBER ===
#'
#' Does the functional visual field radius itself drift between trial 1 and trial 60? The response here is a
#' per-trial FVF estimate (DVA), not a derived coverage count - only estimator D (encircling criterion,
#' `analysis/fvf/fvf.py::estimate_by_encircling`) produces one: it is computed per (subject, trial) directly.
#' Estimator C (selection hazard) needs many pooled opportunities to trace its hazard curve and only ever
#' returns one value per *subject*, so it cannot supply this table - this script is D-only.
#'
#' Mirrors `time_on_task_gam.R`'s structure, but does not reuse `helpers.R::load_data()` - that function is tied
#' to the event-level funnel CSV schema (`is_lws`, `upto_on_target`, ...), which this data does not have.


library(mgcv)
source(file.path("analysis", "R", "helpers.R"))

# set constants
K <- 10

# === Load Data ===
# fvf_radius_by_trial.csv is gitignored and exported by hand from fvf_over_trials.ipynb, with columns:
# subject, trial, fvf_dva
csv_path <- file.path("analysis", "R", "fvf_radius_by_trial.csv")
if (!file.exists(csv_path)) {
  stop(
    "FVF-by-trial results not found at ", csv_path, ".\n",
    "This file is gitignored and exported by hand from fvf_over_trials.ipynb."
  )
}
dat <- read.csv(csv_path)
dat$subject <- as.factor(dat$subject)
dat <- dat[!is.na(dat$fvf_dva), ]


# === Statistical Analysis ===
# fit GAM with trial num as predictor; response is the per-trial FVF radius (DVA) directly, so a plain Gaussian
# GAM - no count/proportion structure to model here, unlike the array-coverage response this replaced.
model <- gam(
  fvf_dva ~ s(trial, k = K, bs = "tp") + s(subject, bs = "re"),
  data = dat,
  method = "REML"
)

# check model results
summary(model)

# diagnostics -> analysis/R/figures/, rather than an anonymous Rplots.pdf in the working directory
plot_path <- open_plot_device("fvf_over_trials_gam_diagnostics.pdf")
gam.check(model)
plot(model, select = 1)
dev.off()
message("diagnostics written to ", plot_path)


# === Export Model Estimates ===
grid <- expand.grid(
  trial = sort(unique(dat$trial)),
  subject = levels(dat$subject)
)

preds <- predict(
  model, newdata = grid, type = "response",
  # exclude = "s(subject)"  # uncomment to calculate the same radius for all subjects (mean subject's FVF)
)
grid$fvf_dva_pred <- preds

# save predictions to file
outfile <- file.path("analysis", "R", "fvf_over_trials_predictions.csv")
write.csv(grid, outfile, row.names = FALSE)
