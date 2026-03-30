
#' === LWS PROBABILITY OVER SPACE ===


library(mgcv)
source(file.path("analysis", "R", "helpers.R"))

# set constants
set.seed(42)
K <- 15


# load and filter the data
dat <- load_data(
  file.path("analysis" ,"R", "funnel_results.csv"),
  valid_only=TRUE,
  on_target_only = TRUE
)


# fit a GAM with only main effects
simple_model <- gam(
  is_lws ~ trial_category +
    te(x, y, k=K, bs="tp") +
    s(subject, bs="re"),
  data = dat,
  family = binomial(),
  method = "REML"
)
gam.check(simple_model)


# fit a GAM with main + interaction effects
interaction_model <- gam(
  is_lws ~ trial_category +
    te(x, y, k=K, bs="tp") +
    te(x, y, k=K, bs="tp", by=trial_category) +
    s(subject, bs="re"),
  data = dat,
  family = binomial(),
  method = "REML"
)
gam.check(interaction_model)


# === STATISTICAL TESTS ===
# (1) check if p[te(...)] < 0.05 to determine if there is a spatial effect on LWS prob.
summary(simple_model)

# (2) check if there's a different spatial pattern for different trial types
anova(simple_model, interaction_model, test = "Chisq")

# (3) post-hoc - check what trial types drove the differences
summary(interaction_model)


# === Export Model Estimates ===
# Create a fine grid for the screen
x_range <- range(dat$x, na.rm = TRUE)
y_range <- range(dat$y, na.rm = TRUE)
grid <- expand.grid(
  x = seq(x_range[1], x_range[2], length.out = 160),
  y = seq(y_range[1], y_range[2], length.out = 90),
  trial_category = levels(dat$trial_category),
  subject = levels(dat$subject)
)

# predict the LWS probability across screen locations, for each subject & trial type
preds <- predict(
  simple_model,
  newdata = grid,
  type = "response",
  # exclude = "s(subject)"  # uncomment to calculate the same probability for all subjects (mean subject's probability)
)
grid$prob <- preds  # append the predicted probability column

# save to file
outfile <- file.path("analysis", "R", "spatial_lws_predictions.csv")
write.csv(grid, outfile, row.names = FALSE)
