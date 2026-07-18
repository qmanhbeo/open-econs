#!/usr/bin/env Rscript
# Ordered logit/probit parity fixtures for open-econs.
# Ground truth = R MASS::polr (Venables & Ripley 2002), the canonical R
# ordered-logit/ordered-probit estimator.
#
# argv[1] = input csv  (tests/r/fixtures/inputs/ordered_input.csv)
# argv[2] = output json (tests/r/fixtures/expected/ordered.json)
#
# Records, for the canonical ordered model (y ~ x1 + x2 + x3, 4 categories):
#   - ologit: coefficients (x1,x2,x3), cutpoints (cut1,cut2,cut3), SEs, logLik
#   - oprobit: same structure with probit link
#
# NOTE on cutpoint sign (rule 16, documented in methodology/limited/ordered.md):
#   polr parameterizes P(Y <= j) = F(c_j - eta) with c_j the stored cutpoint.
#   Stata ologit parameterizes P(Y <= j) = F(eta - cut_j), so Stata's cut_j is
#   the NEGATIVE of polr's c_j. OE stores cutpoints in Stata convention
#   (so they match Stata fixtures directly); the polr->OE conversion is negated.

library(MASS)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
in_csv  <- args[1]
out_json <- args[2]

df <- read.csv(in_csv)
df$y <- ordered(as.factor(df$y), levels = c("0","1","2","3"))

# --- ordered logit ---
fit_logit <- polr(y ~ x1 + x2 + x3, data = df, method = "logistic", Hess = TRUE)
b_logit <- coef(fit_logit)
cut_logit <- fit_logit$zeta
se_logit <- summary(fit_logit)$coefficients[, "Std. Error"]
ll_logit <- logLik(fit_logit)

# --- ordered probit ---
fit_probit <- polr(y ~ x1 + x2 + x3, data = df, method = "probit", Hess = TRUE)
b_probit <- coef(fit_probit)
cut_probit <- fit_probit$zeta
se_probit <- summary(fit_probit)$coefficients[, "Std. Error"]
ll_probit <- logLik(fit_probit)

out <- list(
  # ologit
  ologit_b_x1 = unname(b_logit["x1"]),
  ologit_b_x2 = unname(b_logit["x2"]),
  ologit_b_x3 = unname(b_logit["x3"]),
  ologit_cut1 = unname(cut_logit[1]),
  ologit_cut2 = unname(cut_logit[2]),
  ologit_cut3 = unname(cut_logit[3]),
  ologit_se_x1 = unname(se_logit["x1"]),
  ologit_se_x2 = unname(se_logit["x2"]),
  ologit_se_x3 = unname(se_logit["x3"]),
  ologit_ll = as.numeric(ll_logit),
  # oprobit
  oprobit_b_x1 = unname(b_probit["x1"]),
  oprobit_b_x2 = unname(b_probit["x2"]),
  oprobit_b_x3 = unname(b_probit["x3"]),
  oprobit_cut1 = unname(cut_probit[1]),
  oprobit_cut2 = unname(cut_probit[2]),
  oprobit_cut3 = unname(cut_probit[3]),
  oprobit_se_x1 = unname(se_probit["x1"]),
  oprobit_se_x2 = unname(se_probit["x2"]),
  oprobit_se_x3 = unname(se_probit["x3"]),
  oprobit_ll = as.numeric(ll_probit)
)

cat("writing", out_json, "\n")
write_json(out, out_json, digits = 15, auto_unbox = TRUE, pretty = TRUE)
