
#' === ARRAY COVERAGE OVER TRIAL NUMBER ===
#'
#' Does the fraction of icons within a subject's FVF (array coverage, from `array_coverage.ipynb`) shrink or
#' grow between trial 1 and trial 60? Mirrors `time_on_task_gam.R`'s structure, but the response here is a
#' per-trial count (icons covered out of 180), not a per-visit binary outcome, so it does not reuse
#' `helpers.R::load_data()` - that function is tied to the event-level funnel CSV schema (`is_lws`,
#' `upto_on_target`, ...), which array-coverage data does not have.
#'
#' Companion to `fvf_over_trials_gam.R`, which models the FVF radius itself (only possible for estimator D).
#' This script instead asks a different question - does the *coverage* achieved under a given FVF radius
#' change over the session - and runs for both C and D, since coverage only needs *a* radius per trial, not
#' a per-trial *estimate* of one (see `fvf_coverage_over_trials.ipynb` for why that distinction matters for D).


library(mgcv)
source(file.path("analysis", "R", "helpers.R"))

# set constants
K <- 10

# optional suffix (e.g. "Rscript fvf_coverage_over_trials_gam.R D") to run against a differently-named coverage
# table, for FVF estimators besides the default selection-hazard one - e.g. array_coverage_results_D.csv for D
args <- commandArgs(trailingOnly = TRUE)
suffix <- if (length(args) >= 1) paste0("_", args[1]) else ""

# === Load Data ===
# array_coverage_results{suffix}.csv is gitignored and exported by hand from fvf_coverage_over_trials.ipynb,
# with columns: subject, trial, n_icons, n_covered, coverage_pct
csv_path <- file.path("analysis", "R", paste0("array_coverage_results", suffix, ".csv"))
if (!file.exists(csv_path)) {
  stop(
    "array coverage results not found at ", csv_path, ".\n",
    "This file is gitignored and exported by hand from fvf_coverage_over_trials.ipynb."
  )
}
dat <- read.csv(csv_path)
dat$subject <- as.factor(dat$subject)


# === Statistical Analysis ===
# fit GAM with trial num as predictor; response is a per-trial proportion (icons covered / n_icons), modeled as
# a binomial count so the variance follows the actual trial size rather than assuming constant variance.
model <- gam(
  cbind(n_covered, n_icons - n_covered) ~ s(trial, k = K, bs = "tp") + s(subject, bs = "re"),
  data = dat,
  family = binomial(),
  method = "REML"
)

# check model results
summary(model)

# diagnostics -> analysis/R/figures/, rather than an anonymous Rplots.pdf in the working directory
plot_path <- open_plot_device(paste0("fvf_coverage_over_trials_gam_diagnostics", suffix, ".pdf"))
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
  # exclude = "s(subject)"  # uncomment to calculate the same coverage for all subjects (mean subject's coverage)
)
grid$coverage_prop <- preds

# save predictions to file
outfile <- file.path("analysis", "R", paste0("fvf_coverage_over_trials_predictions", suffix, ".csv"))
write.csv(grid, outfile, row.names = FALSE)
