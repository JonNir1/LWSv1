
# Shared helpers for the LWS GAM scripts.
# Run from the repo root: the scripts resolve paths with file.path("analysis", "R", ...).


#' Open a Named Graphics Device
#'
#' `gam.check()` and `plot()` in a script run via Rscript otherwise write to an anonymous `Rplots.pdf` in the
#' working directory, which is easy to miss and easy to overwrite.
#'
#' @param filename File name (no directory), written into analysis/R/figures/.
#' @return The full path being written to.
open_plot_device <- function(filename) {
  out_dir <- file.path("analysis", "R", "figures")
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  path <- file.path(out_dir, filename)
  pdf(path, width = 8, height = 6)
  return(path)
}
