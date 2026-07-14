#!/usr/bin/env Rscript
# did_sun_abraham.R - R parity anchor for did_sun_abraham() using fixest::sunab().
#
# Sun & Abraham (2021) interaction-weighted estimator:
# y ~ x + sunab(cohort, time) | entity + time
# ATT is a weighted average of period-cohort interaction coefficients,
# weights = cohort-period cell shares.
# SEs: from full VCE of the regression (cluster-robust by entity).
#
# Uses staggered_did_multi_cohort_input.csv with:
#   Entities 0-4: never-treated
#   Entities 5-9: treated at time 2 (cohort = 2)
#   Entities 10-14: treated at time 3 (cohort = 3)
#   Entities 15-19: treated at time 4 (cohort = 4)
#   Entities 20-24: never-treated
#   Entities 25-29: never-treated
#
# Args: argv[1] = input csv, argv[2] = output json

library(fixest)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
df <- read.csv(args[1])

# Run Sun-Abraham estimation
# Formula: y ~ x + sunab(cohort, time) | entity + time
# sunab() creates period x cohort interaction dummies,
# drops reference period (-1) and never-treated cohort
est <- feols(y ~ x + sunab(cohort, time) | entity + time,
             data = df, cluster = ~entity)

# ATT aggregation
att_agg <- aggregate(est, agg = "att")
period_agg <- aggregate(est, agg = "period")
cohort_agg <- aggregate(est, agg = "cohort")

# R-squared
r_squared <- 1 - est$sigma2 * (est$nobs - est$nparams) /
  (sum((df$y - mean(df$y))^2, na.rm = TRUE) / (est$nobs - 1))

# VCE
V_clustered <- vcov(est)
V_iid <- vcov(est, vcov = "iid")

out <- list(
  N = est$nobs,
  att = unname(att_agg[1, "Estimate"]),
  se = unname(att_agg[1, "Std. Error"]),
  t_stat = unname(att_agg[1, "t value"]),
  p_value = unname(att_agg[1, "Pr(>|t|)"]),
  r_squared = r_squared,
  sigma2 = est$sigma2,
  # VCE matrices
  vce_iid = as.list(c(V_iid)),
  vce_clustered = as.list(c(V_clustered)),
  vce_nrow = nrow(V_clustered),
  vce_ncol = ncol(V_clustered),
  # Raw coefficients (full parameter space, unaggregated, matching VCE dims)
  # coef(est) default on sunab models returns aggregated period-level coefs;
  # coef(est, agg=FALSE) returns the raw interaction-level coefs = object$coefficients
  raw_coef_names = names(coef(est, agg = FALSE)),
  raw_coefficients = as.list(unname(coef(est, agg = FALSE))),
  # Aggregated coefficients (period-level, from summary table)
  coef_names = names(coef(est)),
  coefficients = as.list(unname(coef(est))),
  # Period aggregation
  period_coefs = as.list(unname(period_agg[, "Estimate"])),
  period_ses = as.list(unname(period_agg[, "Std. Error"])),
  period_names = rownames(period_agg),
  # Cohort aggregation
  cohort_coefs = as.list(unname(cohort_agg[, "Estimate"])),
  cohort_ses = as.list(unname(cohort_agg[, "Std. Error"])),
  cohort_names = rownames(cohort_agg),
  # Collinear variables removed
  collin_vars = if (!is.null(est$collin.var)) est$collin.var else character(0),
  # Number of estimated parameters (post collinearity)
  nparams = est$nparams,
  # Number of observations used in estimation
  nobs = est$nobs,
  # Entity and time FE counts (for DOF verification)
  n_entities = length(unique(df$entity)),
  n_times = length(unique(df$time))
)

write_json(out, args[2], auto_unbox = TRUE, digits = 17)
