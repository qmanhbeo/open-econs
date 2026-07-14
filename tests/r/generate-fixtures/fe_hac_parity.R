#!/usr/bin/env Rscript
# fe_hac_parity.R - independent reference for open-econs fe() Newey-West HAC.
#
# Reads a balanced panel CSV (columns: y, x, z, entity, time; entity/time are
# integer 0-based ids) and computes the period-aggregation Newey-West (Driscoll-
# Kraay / Arellano) variance, mirroring open_econs.core.cov.newey_west_cov with
# cluster = time:
#   1. two-way within transform (closed form, balanced panel):
#        y* = y - mean_entity(y) - mean_time(y) + grand_mean(y)
#      applied to y, x, z.
#   2. OLS slopes on demeaned data, NO intercept (the FE absorb it).
#   3. scores s_it = x_it * e_it; aggregated within each period:
#        A_t = sum_{i} s_it   (column sums over entities in period t)
#   4. long-run covariance with Bartlett kernel w_l = 1 - l/(L+1):
#        S = sum_t A_t A_t' + sum_{l=1..L} w_l (sum_t A_t A_{t-l}' + h.c.)
#   5. V = (X'X)^-1 S (X'X)^-1   (raw, before any N/(N-K) or df scaling)
#
# Output JSON: coefficients, std_errors, full covariance, lags, columns.
#
# Args: argv[1] = input csv, argv[2] = output json.

library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
in_csv <- args[1]
out_json <- args[2]

df <- read.csv(in_csv)
L <- 2L

entity <- df$entity
time <- df$time

# Two-way demean (closed form for balanced panels).
demean <- function(v, g) {
  m <- tapply(v, g, mean)
  v - m[as.character(g)]
}
yd <- demean(demean(df$y, entity), time)
xd <- demean(demean(df$x, entity), time)
zd <- demean(demean(df$z, entity), time)

Xd <- cbind(xd, zd)
colnames(Xd) <- c("x", "z")
yd <- as.numeric(yd)
Xd <- as.matrix(Xd)

fit <- lm.fit(Xd, yd)
coef <- as.numeric(fit$coefficients)
fitted <- as.numeric(Xd %*% coef)
resid <- yd - fitted
scores <- Xd * resid  # n x k

k <- ncol(Xd)
tt <- sort(unique(time))
A <- t(sapply(tt, function(tt0) {
  colSums(scores[time == tt0, , drop = FALSE])
}))

S0 <- t(A) %*% A
S <- S0
for (lag in seq_len(L)) {
  w <- 1 - lag / (L + 1)
  Gamma <- matrix(0, nrow = k, ncol = k)
  for (t in seq((lag + 1), nrow(A))) {
    Gamma <- Gamma + t(A[t, , drop = FALSE]) %*% A[t - lag, , drop = FALSE]
  }
  S <- S + w * (Gamma + t(Gamma))
}

XtX_inv <- solve(t(Xd) %*% Xd)
V <- XtX_inv %*% S %*% XtX_inv
se <- sqrt(diag(V))

out <- list(
  coefficients = coef,
  std_errors = as.numeric(se),
  cov = lapply(seq_len(nrow(V)), function(i) as.numeric(V[i, ])),
  lags = L,
  columns = colnames(Xd)
)
write_json(out, out_json, auto_unbox = TRUE, digits = 17)
