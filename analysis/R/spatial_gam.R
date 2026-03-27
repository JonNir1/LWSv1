
library(mgcv)

# set constants
set.seed(42)
K <- 30


# load the data
dat <- read.csv(file.path("analysis" ,"R", "spatial_lws.csv"))
dat$subject <- as.factor(dat$subject)
dat$trial_category <- as.factor(dat$trial_category)
dat$is_lws <- as.numeric(dat$is_lws)


# fit a GAM with only main effects
simple_model <- gam(
  is_lws ~ trial_category +
    te(scaled_x, scaled_y, k=K, bs="tp") +
    s(subject, bs="re"),
  data = dat,
  family = binomial(),
  method = "REML"
)
gam.check(simple_model)


# fit a GAM with main + interaction effects
interaction_model <- gam(
  is_lws ~ trial_category +
    te(scaled_x, scaled_y, k=K, bs="tp") +
    te(scaled_x, scaled_y, k=K, bs="tp", by=trial_category) +
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

