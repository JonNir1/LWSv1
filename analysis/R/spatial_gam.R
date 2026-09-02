
#' === LWS PROBABILITY OVER SPACE ===
#'
#' Does LWS probability vary with where on the screen the target sits, and does that pattern differ by trial type?
#' A second section below asks a narrower version of the same question in polar coordinates: does LWS probability
#' depend on eccentricity (distance from screen center) and/or meridian (angle), independent of an arbitrary 2-D
#' spatial pattern?


library(mgcv)
source(file.path("analysis", "R", "helpers.R"))

# set constants
K <- 8                  # per *marginal* basis in te(), so the tensor holds ~K^2 basis functions
MIN_NEIGHBOURS <- 5     # observations within GRID_MASK_RADIUS_PX needed before a grid cell counts as supported
GRID_MASK_RADIUS_PX <- 100


# load and filter the data
dat <- load_data(
  file.path("analysis", "R", "funnel_results.csv"),
  valid_only = TRUE,
  on_target_only = TRUE
)


# === CARTESIAN MODELS: is LWS probability a function of screen position (x, y)? ===

# fit a GAM with only main effects
# CODE_REVIEW.md M10, fixed: s(trial_uid, bs="re") (trial_uid = subject:trial, built in helpers.R::load_data())
# absorbs the visit-within-trial-within-subject nesting alongside the subject-level random intercept.
#
# bam() (not gam()) because trial_uid has ~1,300 levels: gam()'s dense REML fitting is impractically slow at
# that many random-effect coefficients, while bam(..., discrete=TRUE) is mgcv's own fast-fitting path for
# exactly this case (many rows and/or many RE levels) - same model, same smooth specs, a faster backend.
# method="fREML" is bam's (fast) REML.
simple_model <- bam(
  is_lws ~ trial_category +
    te(x, y, k = K, bs = "tp") +
    s(subject, bs = "re") +
    s(trial_uid, bs = "re"),
  data = dat,
  family = binomial(),
  method = "fREML",
  discrete = TRUE
)

# fit a GAM with main + interaction effects
interaction_model <- bam(
  is_lws ~ trial_category +
    te(x, y, k = K, bs = "tp") +
    te(x, y, k = K, bs = "tp", by = trial_category) +
    s(subject, bs = "re") +
    s(trial_uid, bs = "re"),
  data = dat,
  family = binomial(),
  method = "fREML",
  discrete = TRUE
)

# CODE_REVIEW.md M11, fixed - but not the way first proposed. anova() needs likelihoods that are comparable
# across the two models' differing smooth structure, which fREML fits are not (mgcv itself warns against
# this), so the original fix (refit both under method="ML" for that one comparison) is the textbook answer.
# In practice it isn't reachable here: discrete=TRUE - needed for bam() to fit at all in reasonable time with
# trial_uid's ~1,300 levels (see above) - only works with method="fREML"/"NCV", not "ML" (bam() warns and
# silently falls back to the non-discretized path), and the non-discretized ML refit did not finish in over
# 5 hours of CPU time. Rather than accept that cost, drop the two-model anova() comparison entirely: mgcv's
# `by`-factor smooth already gives a valid within-fit test of "does trial type change the spatial pattern" -
# each level of te(x, y, by=trial_category) is fit and tested (edf, p-value) inside interaction_model's own
# fREML fit, no second model or refit needed. Read those directly off summary(interaction_model) below.

# diagnostics -> analysis/R/figures/, rather than an anonymous Rplots.pdf in the working directory
# CODE_REVIEW.md L6: k.check() reports whether K is large enough for te(x, y) - printed here (not only
# plotted) so it's part of the script's captured output.
print(k.check(simple_model))
print(k.check(interaction_model))
plot_path <- open_plot_device("spatial_gam_diagnostics.pdf")
gam.check(simple_model)
gam.check(interaction_model)
plot(simple_model, select = 1, scheme = 2)
dev.off()
message("diagnostics written to ", plot_path)


# === STATISTICAL TESTS ===
# (1) is there a spatial effect on LWS probability at all?
summary(simple_model)

# (2) does the spatial pattern differ by trial type? a global te(x, y) and a by-factor te(x, y) share basis
# functions, so check how far they are confounded before reading the test below - high concurvity makes the
# attribution between them unreliable
print(concurvity(interaction_model, full = FALSE))

# (3) each by-level smooth's own edf/p-value below IS the test for (2): does trial type's spatial pattern
# deviate from the shared te(x, y) main effect. See the M11 note above for why this replaces a two-model
# anova() comparison.
summary(interaction_model)


# === Export Model Estimates ===
# Create a fine grid for the screen
x_range <- range(dat$x, na.rm = TRUE)
y_range <- range(dat$y, na.rm = TRUE)
grid <- expand.grid(
  x = seq(x_range[1], x_range[2], length.out = 160),
  y = seq(y_range[1], y_range[2], length.out = 90),
  trial_category = levels(dat$trial_category),
  subject = levels(dat$subject),
  trial_uid = levels(dat$trial_uid)[1]  # placeholder level; excluded from the prediction below
)

# predict the LWS probability across screen locations, for each subject & trial type
preds <- predict(
  simple_model,
  newdata = grid,
  type = "response",
  exclude = "s(trial_uid)",
  # exclude = c("s(trial_uid)", "s(subject)")  # uncomment to calculate the same probability for all subjects (mean subject's probability)
)
grid$prob <- preds  # append the predicted probability column

# The grid spans the bounding box of observed gaze, which includes screen regions holding little or no data;
# predictions there are extrapolation and must not be plotted as if they were estimates. Flag how much data supports
# each cell so the plotting code can mask the unsupported ones.
#
# Counts observations in an axis-aligned box of +/- GRID_MASK_RADIUS_PX around each cell, via a summed-area table:
# O(cells + observations) rather than the O(cells * observations) a pairwise distance loop would cost.
x_seq <- sort(unique(grid$x))
y_seq <- sort(unique(grid$y))
nx <- length(x_seq)
ny <- length(y_seq)
dx <- if (nx > 1) x_seq[2] - x_seq[1] else Inf
dy <- if (ny > 1) y_seq[2] - y_seq[1] else Inf
radius_x <- max(0, floor(GRID_MASK_RADIUS_PX / dx))
radius_y <- max(0, floor(GRID_MASK_RADIUS_PX / dy))

observed <- dat[stats::complete.cases(dat[, c("x", "y")]), c("x", "y")]
obs_ix <- pmin(pmax(round((observed$x - x_seq[1]) / dx) + 1, 1), nx)
obs_iy <- pmin(pmax(round((observed$y - y_seq[1]) / dy) + 1, 1), ny)
counts <- matrix(tabulate((obs_iy - 1) * nx + obs_ix, nbins = nx * ny), nrow = nx, ncol = ny)

# summed-area table, padded with a zero row/column so the box lookup needs no special-casing at the edges
sat <- matrix(0, nx + 1, ny + 1)
sat[-1, -1] <- counts
sat <- apply(sat, 2, cumsum)
sat <- t(apply(sat, 1, cumsum))

cell_i <- matrix(seq_len(nx), nx, ny)
cell_j <- matrix(rep(seq_len(ny), each = nx), nx, ny)
lo_x <- pmax(1, cell_i - radius_x); hi_x <- pmin(nx, cell_i + radius_x)
lo_y <- pmax(1, cell_j - radius_y); hi_y <- pmin(ny, cell_j + radius_y)
n_nearby <- sat[cbind(as.vector(hi_x + 1), as.vector(hi_y + 1))] -
  sat[cbind(as.vector(lo_x), as.vector(hi_y + 1))] -
  sat[cbind(as.vector(hi_x + 1), as.vector(lo_y))] +
  sat[cbind(as.vector(lo_x), as.vector(lo_y))]

cells <- data.frame(
  x = x_seq[as.vector(cell_i)],
  y = y_seq[as.vector(cell_j)],
  n_nearby = n_nearby
)
grid <- merge(grid, cells, by = c("x", "y"), all.x = TRUE)
grid$is_supported <- grid$n_nearby >= MIN_NEIGHBOURS
message(
  sprintf(
    "%.1f%% of grid cells have >= %d observations within +/-%d px; the rest are extrapolation (is_supported = FALSE).",
    100 * mean(grid$is_supported), MIN_NEIGHBOURS, GRID_MASK_RADIUS_PX
  )
)

# save to file
outfile <- file.path("analysis", "R", "spatial_lws_predictions.csv")
write.csv(grid, outfile, row.names = FALSE)


# === POLAR / ECCENTRICITY MODEL: is LWS probability a function of distance-from-center and/or angle? ===
#
# te(x, y) above can, in principle, fit an arbitrary 2-D surface, including a radially symmetric one - so it isn't
# that Cartesian coordinates *can't* capture an eccentricity effect, it's that te(x, y) doesn't *isolate* one from
# incidental left/right or top/bottom idiosyncrasies. s(r) + s(theta) is a deliberately narrower, additive model
# (no r*theta interaction): "the effect, if any, depends on distance from center and/or angle, each on its own."
# It spends its (much smaller) degrees of freedom on that specific shape instead of sharing them with an
# unconstrained 2-D surface, so a genuine eccentricity or meridian effect is easier to detect and interpret here
# than in the te(x, y) fit above - and a high-edf result here would be as suspicious as a high-edf te(x, y).
#
# H0: no effect of distance-from-center (r) and no effect of angle (theta) on P[LWS]. Under H0, s(r) and s(theta)
# should each shrink toward a flat (constant) function - edf near 0 - not toward a linear one. A low-but-nonzero
# edf (roughly 1-3) is a simple, credible, interpretable effect; edf close to K is as likely to reflect the model
# fitting noise as a real pattern - see the note in spatial_effects.ipynb for how to read edf here.
#
# CENTER_X/CENTER_Y must match constants.py's TOBII_MONITOR (1920x1080, top-left pixel origin) - kept in raw
# pixels, consistent with the Cartesian model above and with CODE_REVIEW.md M12's caveat that px2deg's
# eccentricity bias would otherwise be baked into the very metric this section is testing.
CENTER_X <- 960
CENTER_Y <- 540
Kr <- 10   # basis size for s(r)
Kth <- 8   # basis size for s(theta); modest, since a meridian effect (if real) is expected to be a coarse pattern

dat$r <- sqrt((dat$x - CENTER_X)^2 + (dat$y - CENTER_Y)^2)                                   # px, distance from screen center
dat$theta <- (atan2(-(dat$y - CENTER_Y), dat$x - CENTER_X) * 180 / pi) %% 360                # deg, 0=right, 90=up, wraps 0/360

# bam(..., discrete=TRUE) rather than gam(), for the same reason as the Cartesian models above: trial_uid's
# ~1,300 levels make gam()'s dense REML fitting impractically slow.
eccentricity_model <- bam(
  is_lws ~ trial_category +
    s(r, k = Kr, bs = "tp") +
    s(theta, k = Kth, bs = "cc") +
    s(subject, bs = "re") +
    s(trial_uid, bs = "re"),
  data = dat,
  family = binomial(),
  method = "fREML",
  discrete = TRUE,
  knots = list(theta = c(0, 360))  # cyclic spline needs an explicit period, or it wraps at data's min/max instead of 0/360
)

summary(eccentricity_model)
print(k.check(eccentricity_model))

plot_path <- open_plot_device("spatial_eccentricity_gam_diagnostics.pdf")
gam.check(eccentricity_model)
plot(eccentricity_model, select = 1)  # s(r)
plot(eccentricity_model, select = 2)  # s(theta)
dev.off()
message("diagnostics written to ", plot_path)

# Two 1-D sweeps rather than a full r x theta cross product: the model is additive (no r*theta interaction), so
# the *shape* of each smooth does not depend on where the other predictor is held - only its vertical offset does.
# `sweep` marks which predictor varies in a given row, for the plotting code to split on.
r_range <- range(dat$r, na.rm = TRUE)
grid_r <- expand.grid(
  r = seq(r_range[1], r_range[2], length.out = 100),
  theta = median(dat$theta),
  trial_category = levels(dat$trial_category),
  subject = levels(dat$subject),
  trial_uid = levels(dat$trial_uid)[1]
)
grid_r$sweep <- "r"
grid_theta <- expand.grid(
  r = median(dat$r),
  theta = seq(0, 360, length.out = 72),
  trial_category = levels(dat$trial_category),
  subject = levels(dat$subject),
  trial_uid = levels(dat$trial_uid)[1]
)
grid_theta$sweep <- "theta"
grid_ecc <- rbind(grid_r, grid_theta)

outfile_ecc <- file.path("analysis", "R", "spatial_eccentricity_lws_predictions.csv")
predict_and_export(eccentricity_model, grid_ecc, outfile_ecc, exclude = "s(trial_uid)")
