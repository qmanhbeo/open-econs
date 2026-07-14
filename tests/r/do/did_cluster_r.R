#!/usr/bin/env Rscript
# did_cluster_r.R - R parity anchor for did() cluster SEs using fixest::feols.
#
# Two-period DiD with entity-level cluster SEs.
# Matches Stata: regress y treat post treat_post, cluster(unit)
#
# Args: argv[1] = input csv, argv[2] = output json

library(fixest)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
df <- read.csv(args[1])
df$treat_post <- df$treat * df$post

m <- feols(y ~ treat + post + treat_post, data = df, cluster = "unit")

out <- list(
  se_treatXpost = sqrt(vcov(m)["treat_post", "treat_post"])
)
write_json(out, args[2], auto_unbox = TRUE, digits = 17)
