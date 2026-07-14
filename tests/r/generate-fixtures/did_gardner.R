#!/usr/bin/env Rscript
# did_gardner.R - R parity anchor for did_gardner() using did2s::did2s().
#
# Gardner (2022) Two-Stage DID: first-stage regression on untreated units,
# second-stage regression of residuals on treatment indicator.
# SEs: cluster-robust (by entity).
#
# Uses did_cs_balanced_input.csv with treatment constructed as:
# entities 10-19 treated at time >= 3, entities 0-9 never-treated.
#
# Args: argv[1] = input csv, argv[2] = output json

library(did2s)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
df <- read.csv(args[1])

# Construct treatment indicator: entities 10-19 treated at time >= 3
df$treat <- as.integer(df$entity >= 10 & df$entity < 20 & df$time >= 3)

# Run Gardner DID2S
# First stage: entity FE + time FE, estimated on untreated only
# Second stage: treatment indicator on all units
est <- did2s(df, yname = "y",
             first_stage = ~ 0 + factor(entity) + factor(time),
             second_stage = ~ treat,
             treatment = "treat",
             cluster_var = "entity",
             verbose = FALSE)

# Extract results
ct <- summary(est)$coeftable

# Compute R-squared from SSR
r_squared <- 1 - est$ssr / est$ssr_null
adj_r_squared <- 1 - (est$ssr / (est$nobs - est$nparams)) / (est$ssr_null / (est$nobs - 1))

out <- list(
  N = est$nobs,
  att = unname(coef(est)["treat"]),
  se = unname(est$se["treat"]),
  t_stat = unname(ct["treat", "t value"]),
  p_value = unname(ct["treat", "Pr(>|t|)"]),
  r_squared = r_squared,
  adj_r_squared = adj_r_squared,
  sigma2 = est$sigma2,
  # VCE matrices
  vce_iid = as.list(c(est$cov.iid)),
  vce_clustered = as.list(c(est$cov.scaled)),
  # Coefficient names
  coef_names = names(est$coefficients)
)

write_json(out, args[2], auto_unbox = TRUE, digits = 17)
