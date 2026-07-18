#!/usr/bin/env Rscript
# Poisson FE (PPML) parity fixture for open-econs.
# Ground truth = R fixest::fepois (Berge 2018), the Python pyfixest port's source.
#
# argv[1] = input csv  (tests/r/fixtures/inputs/poisson_input.csv)
# argv[2] = output json (tests/r/fixtures/expected/poisson.json)
#
# Records BOTH conventions (rule 15 toggle):
#   - "fixest" : fepois default ssc  -> matches oe.poisson vcov_backend="fixest"
#   - "stata"  : ssc(fixef.K="none", adj=FALSE, cluster.adj=TRUE)
#                -> matches oe.poisson vcov_backend="stata" (== Stata ppmlhdfe)
# For each: coefs, cluster(firm) SEs, iid SEs, deviance, logLik, pseudo-R2.

library(fixest)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
in_csv  <- args[1]
out_json <- args[2]

df <- read.csv(in_csv)

mod_clu <- fepois(y ~ x1 + x2 | firm + year, data = df, cluster = ~firm)

# fixest-default SEs (oe vcov_backend="fixest")
mod_clu_fixest <- summary(mod_clu)
se_x1_fixest <- as.numeric(se(mod_clu_fixest)["x1"])
se_x2_fixest <- as.numeric(se(mod_clu_fixest)["x2"])

# stata-equivalent SEs (oe vcov_backend="stata")
mod_clu_stata <- summary(mod_clu, ssc = ssc(fixef.K = "none", adj = FALSE, cluster.adj = TRUE))
se_x1_stata <- as.numeric(se(mod_clu_stata)["x1"])
se_x2_stata <- as.numeric(se(mod_clu_stata)["x2"])

# iid SEs
mod_iid <- summary(fepois(y ~ x1 + x2 | firm + year, data = df))
se_x1_iid <- as.numeric(se(mod_iid)["x1"])
se_x2_iid <- as.numeric(se(mod_iid)["x2"])

out <- list(
  b_x1      = unname(coef(mod_clu)["x1"]),
  b_x2      = unname(coef(mod_clu)["x2"]),
  se_x1_fixest = se_x1_fixest,
  se_x2_fixest = se_x2_fixest,
  se_x1_stata  = se_x1_stata,
  se_x2_stata  = se_x2_stata,
  se_x1_iid  = se_x1_iid,
  se_x2_iid  = se_x2_iid,
  deviance   = as.numeric(deviance(mod_clu)),
  loglik     = as.numeric(logLik(mod_clu)),
  # fixest's McFadden pseudo-R2 is the 4th element of r2(); this is exactly the
  # quantity pyfixest exposes as _pseudo_r2 (and oe.poisson reports as pseudo_r2).
  pseudo_r2  = as.numeric(r2(mod_clu)[4])
)

cat("writing", out_json, "\n")
write_json(out, out_json, digits = 15, auto_unbox = TRUE, pretty = TRUE)
