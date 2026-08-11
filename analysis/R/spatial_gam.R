
#' === LWS PROBABILITY OVER SPACE ===
#'
#' Does LWS probability vary with where on the screen the target sits, and does that pattern differ by trial type?


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


# fit a GAM with only main effects
# WARNING (CODE_REVIEW.md M11, unresolved): both models are fitted with REML, but they are compared with anova()
# below. REML log-likelihoods are not comparable across models differing in their fixed/smooth structure, so that
# comparison is not valid as it stands. Refit both with method = "ML" for the comparison, keeping REML for whichever
# model is finally reported. Left as-is pending a decision.
simple_model <- gam(
  is_lws ~ trial_category +
    te(x, y, k = K, bs = "tp") +
    s(subject, bs = "re"),
  data = dat,
  family = binomial(),
  method = "REML"
)

# fit a GAM with main + interaction effects
interaction_model <- gam(
  is_lws ~ trial_category +
    te(x, y, k = K, bs = "tp") +
    te(x, y, k = K, bs = "tp", by = trial_category) +
    s(subject, bs = "re"),
  data = dat,
  family = binomial(),
  method = "REML"
)

# diagnostics -> analysis/R/figures/, rather than an anonymous Rplots.pdf in the working directory
plot_path <- open_plot_device("spatial_gam_diagnostics.pdf")
gam.check(simple_model)
gam.check(interaction_model)
plot(simple_model, select = 1, scheme = 2)
dev.off()
message("diagnostics written to ", plot_path)


# === STATISTICAL TESTS ===
# (1) is there a spatial effect on LWS probability at all?
summary(simple_model)

# (2) does the spatial pattern differ by trial type?
# a global te(x, y) and a by-factor te(x, y) share basis functions, so check how far they are confounded before
# reading the comparison - high concurvity makes the attribution between them unreliable
print(concurvity(interaction_model, full = FALSE))
anova(simple_model, interaction_model, test = "Chisq")   # see the M11 warning above: not valid under REML

# (3) post-hoc - which trial types drove the difference
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
