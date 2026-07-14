#!/usr/bin/env Rscript
# did_basic_r.R - R parity anchor for did() using fixest::feols (plain OLS).
#
# Two-period DiD: y ~ treat + post + treat_post, no HC, no cluster.
# Matches Stata: regress y treat post treat_post
#
# Args: argv[1] = input csv, argv[2] = output json

library(fixest)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
df <- read.csv(args[1])
df$treat_post <- df$treat * df$post

m <- feols(y ~ treat + post + treat_post, data = df)

out <- list(
  N = nobs(m),
  b_int = coef(m)["(Intercept)"],
  b_treatXpost = coef(m)["treat_post"],
  se_treatXpost = sqrt(vcov(m)["treat_post", "treat_post"])
)
write_json(out, args[2], auto_unbox = TRUE, digits = 17)
