#!/usr/bin/env Rscript
# qreg.R - Genuine R-parity fixture for open-econs quantile_reg().
#
# Uses R's quantreg package (rq(method="br")) as the independent reference for
# coefficients and the kernel sandwich SE (summary.rq(se="ker", hs=TRUE)).
#
# Conventions:
#   [NATIVE]  = taken directly from R package output (rq coef, summary SE)
#
# Key findings applied here:
#   - R quantreg::rq(method="br") is the Barrodale-Roberts simplex; matches
#     Stata qreg AND open-econs LP solve to machine precision.
#   - open-econs se_method="ker" reproduces R's summary.rq(se="ker", hs=TRUE)
#     (Powell kernel, Hall-Sheather bandwidth rescaled by residual spread) to
#     <=1e-6.  This is the R-parity target.
#   - open-econs se_method="stata" reproduces *Stata* (handled by the Stata
#     fixture); it intentionally differs from R's kernel SE, so it is NOT
#     asserted here.
#
# Args: argv[1] = input csv, argv[2] = output json.

library(quantreg)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
in_csv <- args[1]
out_json <- args[2]

df <- read.csv(in_csv)

fit_q50 <- rq(y ~ x1 + x2, data = df, tau = 0.5, method = "br")
fit_q25 <- rq(y ~ x1 + x2, data = df, tau = 0.25, method = "br")

# Summary with Powell kernel, Hall-Sheather bandwidth [NATIVE R]
sum_q50 <- summary(fit_q50, se = "ker", hs = TRUE)
sum_q25 <- summary(fit_q25, se = "ker", hs = TRUE)

out <- list()
out$q50 <- list(
  coef = as.numeric(coef(fit_q50)),
  se   = as.numeric(sum_q50$coefficients[, 2]),
  tau  = 0.5
)
out$q25 <- list(
  coef = as.numeric(coef(fit_q25)),
  se   = as.numeric(sum_q25$coefficients[, 2]),
  tau  = 0.25
)

write_json(out, out_json, auto_unbox = TRUE, digits = 15)

cat("=== R qreg Fixture (quantreg) ===\n")
cat("tau=0.5  coef:", out$q50$coef, "\n")
cat("tau=0.5  kerSE:", out$q50$se, "\n")
cat("tau=0.25 coef:", out$q25$coef, "\n")
cat("tau=0.25 kerSE:", out$q25$se, "\n")
