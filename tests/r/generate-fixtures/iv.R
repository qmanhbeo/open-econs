#! iv.R -- R (AER::ivreg + sandwich) ground truth for open_econs iv() parity.
# Conventions captured (source-confirmed 2026-07-17):
#   * Coefficient: 2SLS from AER::ivreg(y ~ w + x | w + z1 + z2).  Variables
#     before `|` are ALL treated as endogenous regressors; repeating `w` on
#     both sides instruments the exogenous `w` with itself, so the estimate
#     equals OE `iv(y ~ w | x ~ z1 + z2)`.
#   * nonrobust: AER::ivreg vcov() => s2 = SSR/(N-K)  (debiased; matches OE
#     cov_type="nonrobust", debiased=True).
#   * HC1: sandwich::vcovHC(fit, type="HC1")  (matches OE cov_type="HC1").
#   * cluster: sandwich::vcovCL(fit, cluster=~id, type="HC1") => applies the
#     G/(G-1) SSC (matches OE cluster, debiased=True).
args <- commandArgs(trailingOnly = TRUE)
in_csv  <- args[1]
out_json <- args[2]

library(AER)
library(sandwich)
library(jsonlite)

df <- read.csv(in_csv)
fit <- ivreg(y ~ w + x | w + z1 + z2, data = df)

cf <- as.numeric(coef(fit))
names(cf) <- NULL

se_nr <- as.numeric(sqrt(diag(vcov(fit))))
names(se_nr) <- NULL

se_hc1 <- as.numeric(sqrt(diag(vcovHC(fit, type = "HC1"))))
names(se_hc1) <- NULL

se_cl <- as.numeric(sqrt(diag(vcovCL(fit, cluster = ~id, type = "HC1"))))
names(se_cl) <- NULL

out <- list(
  coef = cf,
  nonrobust = list(coef = cf, se = se_nr),
  hc1 = list(coef = cf, se = se_hc1),
  cluster = list(coef = cf, se = se_cl)
)

write_json(out, out_json, auto_unbox = TRUE, digits = 15, pretty = TRUE)
cat("iv.R: wrote", out_json, "\n")
