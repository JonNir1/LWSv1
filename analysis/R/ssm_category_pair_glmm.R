#' === HIT-CATEGORY x MISS-CATEGORY EFFECT ON LWS PROBABILITY (Q6) ===
#'
#' Sourced via rpy2 (analysis/helpers/r_bridge.py::source_r), not run via Rscript - no CSV I/O here. The
#' caller builds `dat` (subject, trial, hit_category, miss_category, is_lws), restricted to
#' `num_targets_found_before == 1` AND `sufficient_n` cells only (see ssm.py::build_category_pair_table -
#' N >= 10 visits per (hit_category, miss_category) cell), via r_bridge.py::to_r_dataframe() before sourcing
#' this file. Main effects only - no hit_category:miss_category interaction (36 cells is not supportable
#' at this sample size, see the plan doc).

library(lme4)

GLMER_CONTROL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

model_flat <- glmer(
  is_lws ~ hit_category + miss_category + (1 | subject),
  data = dat, family = binomial(), control = GLMER_CONTROL
)
model_nested <- glmer(
  is_lws ~ hit_category + miss_category + (1 | subject/trial),
  data = dat, family = binomial(), control = GLMER_CONTROL
)
