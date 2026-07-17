#! iv.R -- R (AER::ivreg + sandwich) ground truth for open_econs iv() parity.
# Conventions captured (source-confirmed 2026-07-17, HC0/HC2/HC3 added 2026-07-17):
#   * Coefficient: 2SLS from AER::ivreg(y ~ w + x | w + z1 + z2).  Variables
#     before `|` are ALL treated as endogenous regressors; repeating `w` on
#     both sides instruments the exogenous `w` with itself, so the estimate
#     equals OE `iv(y ~ w | x ~ z1 + z2)`.
#   * nonrobust: AER::ivreg vcov() => s2 = SSR/(N-K)  (debiased; matches OE
#     cov_type="nonrobust", debiased=True).
#   * HC0/HC1/HC2/HC3: sandwich::vcovHC(fit, type="HCk").  HCk use the
#     instrument-projected regressors in the meat (AER::estfun.ivreg uses
#     model.matrix(component="projected")).  OE reproduces these to <=1e-6 via
#     the hand-rolled MacKinnon-White sandwich (_iv_hc_sandwich).  HC1 matches
#     OE cov_type="HC1" (debiased=True); HC0/HC2/HC3 match OE with debiased=True
#     (R applies no extra SSR df scale to them beyond the HCk leverage weights).
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

se_hc0 <- as.numeric(sqrt(diag(vcovHC(fit, type = "HC0"))))
names(se_hc0) <- NULL

se_hc2 <- as.numeric(sqrt(diag(vcovHC(fit, type = "HC2"))))
names(se_hc2) <- NULL

se_hc3 <- as.numeric(sqrt(diag(vcovHC(fit, type = "HC3"))))
names(se_hc3) <- NULL

se_cl <- as.numeric(sqrt(diag(vcovCL(fit, cluster = ~id, type = "HC1"))))
names(se_cl) <- NULL

out <- list(
  coef = cf,
  nonrobust = list(coef = cf, se = se_nr),
  hc0 = list(coef = cf, se = se_hc0),
  hc1 = list(coef = cf, se = se_hc1),
  hc2 = list(coef = cf, se = se_hc2),
  hc3 = list(coef = cf, se = se_hc3),
  cluster = list(coef = cf, se = se_cl)
)

write_json(out, out_json, auto_unbox = TRUE, digits = 15, pretty = TRUE)
cat("iv.R: wrote", out_json, "\n")
