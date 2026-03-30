
#' Load and Clean LWS-Funnel Data
#'
#' Reads the funnel results CSV, handles Python-to-R boolean conversion, 
#' casts variables to correct types, and applies experimental filters.
#'
#' @param csv_path Path to the funnel_results.csv file.
#' @param valid_only Logical; if TRUE, keeps only rows where is_valid_trial is TRUE.
#' @param on_target_only Logical; if TRUE, keeps only rows where on_target is TRUE.
#' @return A cleaned data frame ready for GAM fitting.
load_data <- function(csv_path, valid_only = TRUE, on_target_only = TRUE) {
  
  dat <- read.csv(csv_path)
  
  # Convert Python "True"/"False" strings to R logicals
  dat[dat == "True"] <- TRUE
  dat[dat == "False"] <- FALSE
  dat[] <- lapply(dat, type.convert, as.is = TRUE)  # cast column type to logical
  
  # Cast to factor for specific columns:
  dat$subject <- as.factor(dat$subject)
  dat$trial_category <- as.factor(dat$trial_category)
  dat$target_category <- as.factor(dat$target_category)
  dat$is_lws <- as.numeric(dat$is_lws)  # ensure the response column is 0/1 for binomial GAM
  
  # Apply filters based on flags
  if (valid_only) { dat <- subset(dat, is_valid_trial)  }
  if (on_target_only) { dat <- subset(dat, on_target)  }
  
  return(dat)
}
