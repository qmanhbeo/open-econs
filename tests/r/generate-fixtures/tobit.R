#!/usr/bin/env Rscript
# Tobit (censored normal) MLE parity fixture for open-econs.
# Ground truth = R AER::tobit (the project's primary R reference for Tobit;
# censReg is NOT installed on the dev box).
#
# argv[1] = input csv  (tests/r/fixtures/inputs/tobit_input.csv)
# argv[2] = output json (tests/r/fixtures/expected/tobit.json)
#
# Records, for both a left-censored (y ~ x1 x2 x3, left = 0) and a
# no-censoring variant (y_nocens ~ x1 x2 x3):
#   - coefficient point estimates b_x1, b_x2, b_x3
#   - sigma  (AER::tobit reports sigma directly via summary()$sigma)
#   - logLik
#   - nonrobust (OIM) SEs: se_x1, se_x2, se_x3
#   - left-censored count (n_left) via sum(y <= 0)

library(AER)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
in_csv  <- args[1]
out_json <- args[2]

df <- read.csv(in_csv)

# ---- left-censored Tobit at 0 ----
m <- tobit(y_left ~ x1 + x2 + x3, data = df, left = 0, right = Inf)
s <- summary(m)
b_x1 <- unname(coef(s)["x1", "Estimate"])
b_x2 <- unname(coef(s)["x2", "Estimate"])
b_x3 <- unname(coef(s)["x3", "Estimate"])
se_x1 <- unname(coef(s)["x1", "Std. Error"])
se_x2 <- unname(coef(s)["x2", "Std. Error"])
se_x3 <- unname(coef(s)["x3", "Std. Error"])
sigma_l <- as.numeric(s$sigma)
ll_l <- as.numeric(logLik(m))
n_left_l <- sum(df$y_left <= 0)

# ---- no-censoring variant (OLS-equivalent MLE) ----
m2 <- tobit(y_nocens ~ x1 + x2 + x3, data = df, left = -Inf, right = Inf)
s2 <- summary(m2)
b2_x1 <- unname(coef(s2)["x1", "Estimate"])
b2_x2 <- unname(coef(s2)["x2", "Estimate"])
b2_x3 <- unname(coef(s2)["x3", "Estimate"])
se2_x1 <- unname(coef(s2)["x1", "Std. Error"])
se2_x2 <- unname(coef(s2)["x2", "Std. Error"])
se2_x3 <- unname(coef(s2)["x3", "Std. Error"])
sigma2 <- as.numeric(s2$sigma)
ll2 <- as.numeric(logLik(m2))

out <- list(
  b_x1 = b_x1, b_x2 = b_x2, b_x3 = b_x3,
  se_x1 = se_x1, se_x2 = se_x2, se_x3 = se_x3,
  sigma = sigma_l,
  loglik = ll_l,
  n_left = n_left_l,
  b2_x1 = b2_x1, b2_x2 = b2_x2, b2_x3 = b2_x3,
  se2_x1 = se2_x1, se2_x2 = se2_x2, se2_x3 = se2_x3,
  sigma2 = sigma2,
  loglik2 = ll2
)

cat("writing", out_json, "\n")
write_json(out, out_json, digits = 15, auto_unbox = TRUE, pretty = TRUE)
