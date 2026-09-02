
#' === LWS PROBABILITY OVER TRIAL NUMBER ===


library(mgcv)
source(file.path("analysis", "R", "helpers.R"))

# set constants
K <- 10

# load and filter the data
dat <- load_data(
  file.path("analysis", "R", "funnel_results.csv"),
  valid_only = TRUE,
  on_target_only = TRUE
)


# === Statistical Analysis ===
# fit GAM with trial num as predictor
# CODE_REVIEW.md M10, fixed: the unit of observation is a visit, and visits nest within trial within subject.
# s(trial_uid, bs="re") (trial_uid = subject:trial, built in helpers.R::load_data()) absorbs that nesting
# alongside the subject-level random intercept, so the smooth's p-value is no longer anti-conservative.
#
# bam() (not gam()) because trial_uid has ~1,300 levels: gam()'s dense REML fitting is impractically slow at
# that many random-effect coefficients, while bam(..., discrete=TRUE) is mgcv's own fast-fitting path for
# exactly this case (many rows and/or many RE levels) - same model, same smooth specs, a faster backend.
# method="fREML" is bam's (fast) REML.
model <- bam(
  is_lws ~ trial_category + s(trial, k = K, bs = "tp") + s(subject, bs = "re") + s(trial_uid, bs = "re"),
  data = dat,
  family = binomial(),
  method = "fREML",
  discrete = TRUE
)

# check model results
summary(model)

# diagnostics -> analysis/R/figures/, rather than an anonymous Rplots.pdf in the working directory
# CODE_REVIEW.md L6: k.check() reports whether K is large enough for s(trial) - printed here (not only plotted)
# so it's part of the script's captured output.
print(k.check(model))
plot_path <- open_plot_device("time_on_task_gam_diagnostics.pdf")
gam.check(model)
plot(model, select = 1)
dev.off()
message("diagnostics written to ", plot_path)


# === Export Model Estimates ===
grid <- expand.grid(
  trial = sort(unique(dat$trial)),
  trial_category = levels(dat$trial_category),
  subject = levels(dat$subject),
  trial_uid = levels(dat$trial_uid)[1]  # placeholder level; excluded from the prediction below
)

# marginalize probabilities over subjects and trial types:
# final_trend <- aggregate(prob ~ trial, data = grid, FUN = mean)
# plot(final_trend)

# save predictions to file
outfile <- file.path("analysis", "R", "time-on-task_lws_predictions.csv")
predict_and_export(model, grid, outfile, exclude = "s(trial_uid)")
