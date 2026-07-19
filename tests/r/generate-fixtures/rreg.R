# Robust regression (MASS::rlm) parity fixture for open-econs robust_reg().
# Ground truth = R:
#   MASS::rlm(y ~ x1 + x2, method="MM", psi=psi.bisquare, init="ls",
#             scale.est="MAD", maxit=20)   (the "mm" default)
#   MASS::rlm(y ~ x1 + x2, method="M",  psi=psi.bisquare, scale.est="MAD")
#             for the "huber" (plain M) branch.
#
# argv[1] = input csv  (tests/r/fixtures/inputs/rreg_input.csv)
# argv[2] = output json (tests/r/fixtures/expected/rreg.json)
#
# Records (coefficients + MASS::rlm covariance SEs, the "rlm" vcov branch):
#   mm:  b0,b1,b2, se0,se1,se2, scale, w0..wN
#   m:   b0_m,b1_m,b2_m, se0_m,se1_m,se2_m

library(MASS); library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
in_csv  <- args[1]
out_json <- args[2]

df <- read.csv(in_csv)

fit_mm <- rlm(y ~ x1 + x2, data = df, method = "MM", psi = psi.bisquare,
              init = "ls", scale.est = "MAD", maxit = 20)
b0 <- unname(coef(fit_mm)[1]); b1 <- unname(coef(fit_mm)[2]); b2 <- unname(coef(fit_mm)[3])
Vmm <- summary(fit_mm)$cov.unscaled * fit_mm$s^2
se0 <- sqrt(Vmm[1,1]); se1 <- sqrt(Vmm[2,2]); se2 <- sqrt(Vmm[3,3])
scale_mm <- as.numeric(fit_mm$s)
w_mm <- as.numeric(fit_mm$w)

fit_m <- rlm(y ~ x1 + x2, data = df, method = "M", psi = psi.bisquare,
             scale.est = "MAD", maxit = 20)
b0_m <- unname(coef(fit_m)[1]); b1_m <- unname(coef(fit_m)[2]); b2_m <- unname(coef(fit_m)[3])
Vm <- summary(fit_m)$cov.unscaled * fit_m$s^2
se0_m <- sqrt(Vm[1,1]); se1_m <- sqrt(Vm[2,2]); se2_m <- sqrt(Vm[3,3])

out <- list(
  b0 = b0, b1 = b1, b2 = b2,
  se0 = se0, se1 = se1, se2 = se2,
  scale = scale_mm,
  w = w_mm,
  b0_m = b0_m, b1_m = b1_m, b2_m = b2_m,
  se0_m = se0_m, se1_m = se1_m, se2_m = se2_m
)
write_json(out, out_json, digits = 15, auto_unbox = TRUE, pretty = TRUE)
