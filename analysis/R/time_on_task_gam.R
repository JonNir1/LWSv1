
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
# NOTE (CODE_REVIEW.md M10, unresolved): the unit of observation is a visit, but visits nest within target within
# trial within subject. A subject-level random intercept does not absorb the within-trial dependence, so the smooth's
# p-values are anti-conservative. Point estimates are unaffected.
model <- gam(
  is_lws ~ trial_category + s(trial, k = K, bs = "tp") + s(subject, bs = "re"),
  data = dat,
  family = binomial(),
  method = "REML"
)

# check model results
summary(model)

# diagnostics -> analysis/R/figures/, rather than an anonymous Rplots.pdf in the working directory
plot_path <- open_plot_device("time_on_task_gam_diagnostics.pdf")
gam.check(model)
plot(model, select = 1)
dev.off()
message("diagnostics written to ", plot_path)


# === Export Model Estimates ===
grid <- expand.grid(
  trial = sort(unique(dat$trial)),
  trial_category = levels(dat$trial_category),
  subject = levels(dat$subject)
)

preds <- predict(
  model, newdata = grid, type = "response",
  # exclude = "s(subject)"  # uncomment to calculate the same probability for all subjects (mean subject's probability)
)
grid$prob <- preds

# marginalize probabilities over subjects and trial types:
# final_trend <- aggregate(prob ~ trial, data = grid, FUN = mean)
# plot(final_trend)

# save predictions to file
outfile <- file.path("analysis", "R", "time-on-task_lws_predictions.csv")
write.csv(grid, outfile, row.names = FALSE)
