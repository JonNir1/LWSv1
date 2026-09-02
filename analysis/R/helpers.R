
# Shared helpers for the LWS GAM scripts.
# Run from the repo root: the scripts resolve paths with file.path("analysis", "R", ...).


#' Boolean Columns Exported by the Python Funnel
#'
#' `funnel_results.csv` is written by pandas, so every boolean arrives as the literal string "True"/"False".
#' Listing the columns explicitly is safer than scanning the whole frame for those strings, which would also
#' convert a genuine text column that happened to contain "True".
#'
#' The `upto_` prefix marks a *cumulative* funnel column: "passed this criterion and every criterion before it".
#' `upto_on_target` therefore means "in a valid trial AND on target", not "on target". The three unprefixed
#' columns are conjunctions by definition.
FUNNEL_LOGICAL_COLUMNS <- c(
  # trial-level criteria
  "upto_gaze_coverage", "upto_fixation_rate", "upto_has_actions", "upto_no_bad_action",
  "upto_no_miss_with_false_alarm", "is_valid_trial",
  # event-level criteria
  "upto_on_target", "upto_before_identification", "upto_not_close_to_trial_end",
  "upto_not_before_exemplar_visit", "is_lws",
  "upto_after_identification", "is_target_return"
)


#' Convert Python-Exported Booleans to R Logicals
#'
#' @param dat A data frame read from the funnel CSV.
#' @param columns Character vector of columns to convert; missing ones are ignored.
#' @return `dat` with those columns as logical.
as_logical_columns <- function(dat, columns = FUNNEL_LOGICAL_COLUMNS) {
  present <- intersect(columns, names(dat))
  for (col in present) {
    values <- dat[[col]]
    if (is.logical(values)) next
    dat[[col]] <- values %in% c("True", "TRUE", "true")
  }
  return(dat)
}


#' Load and Clean LWS-Funnel Data
#'
#' Reads the funnel results CSV, converts the Python booleans, casts grouping variables to factors, and applies
#' the experimental filters.
#'
#' NOTE: funnel columns are cumulative, so `upto_on_target` already implies `is_valid_trial`. `valid_only` is
#' therefore redundant whenever `on_target_only` is TRUE; both are kept so the intent of a given analysis stays
#' explicit in the calling script.
#'
#' @param csv_path Path to the funnel_results.csv file.
#' @param valid_only Logical; if TRUE, keeps only rows where is_valid_trial is TRUE.
#' @param on_target_only Logical; if TRUE, keeps only rows where upto_on_target is TRUE.
#' @return A cleaned data frame ready for GAM fitting.
load_data <- function(csv_path, valid_only = TRUE, on_target_only = TRUE) {

  if (!file.exists(csv_path)) {
    stop(
      "funnel results not found at ", csv_path, ".\n",
      "This file is gitignored and exported by hand from an analysis notebook - build a funnel and write it there."
    )
  }
  dat <- read.csv(csv_path)
  dat <- as_logical_columns(dat)

  # Cast to factor for specific columns:
  dat$subject <- as.factor(dat$subject)
  dat$trial_category <- as.factor(dat$trial_category)
  dat$target_category <- as.factor(dat$target_category)
  dat$is_lws <- as.numeric(dat$is_lws)  # ensure the response column is 0/1 for binomial GAM

  # Visits nest within trial within subject (CODE_REVIEW.md M10): a subject-only random intercept doesn't
  # absorb that nesting, so smooth p-values come out anti-conservative. trial_uid gives every (subject, trial)
  # pair its own random-intercept level, usable as s(trial_uid, bs = "re") alongside s(subject, bs = "re").
  dat$trial_uid <- interaction(dat$subject, dat$trial, drop = TRUE)

  # Apply filters based on flags
  required <- c(if (valid_only) "is_valid_trial", if (on_target_only) "upto_on_target")
  missing <- setdiff(required, names(dat))
  if (length(missing) > 0) {
    stop(
      "funnel column(s) not found: ", paste(missing, collapse = ", "), ".\n",
      "Funnel columns carry an 'upto_' prefix (CODE_REVIEW.md M7); a CSV exported before that change will use the ",
      "bare criterion names and must be re-exported."
    )
  }
  if (valid_only) { dat <- subset(dat, is_valid_trial) }
  if (on_target_only) { dat <- subset(dat, upto_on_target) }

  return(dat)
}


#' Predict Over a Grid and Export to CSV
#'
#' Wraps the predict() -> append `prob` column -> write.csv() sequence shared by every *_gam.R script's
#' "Export Model Estimates" section.
#'
#' @param model A fitted gam object.
#' @param grid A data frame of predictor combinations to predict over.
#' @param outfile Path to write the resulting CSV to.
#' @param exclude Optional character vector passed through to predict()'s `exclude` (e.g. "s(subject)" to
#'   marginalize out the subject random effect).
#' @return `grid` with the appended `prob` column, invisibly.
predict_and_export <- function(model, grid, outfile, exclude = NULL) {
  grid$prob <- predict(model, newdata = grid, type = "response", exclude = exclude)
  write.csv(grid, outfile, row.names = FALSE)
  invisible(grid)
}


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
