
#' === LWS PROBABILITY OVER TRIAL NUMBER ===


library(mgcv)
source(file.path("analysis", "R", "helpers.R"))

# set constants
set.seed(42)

# load and filter the data
dat <- load_data(
  file.path("analysis" ,"R", "funnel_results.csv"),
  valid_only=TRUE,
  on_target_only = TRUE
)


# === Statistical Analysis ===
# fit GAM with trial num as predictor
K <- 10

model <- gam(
  is_lws ~ trial_category + s(trial, k=K, bs="tp") + s(subject, bs="re"),
  data = dat,
  family = binomial(),
  method = "REML"
)
gam.check(model)

# check model results
summary(model)
plot(model, select = 1)


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
