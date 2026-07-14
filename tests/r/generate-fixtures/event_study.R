#!/usr/bin/env Rscript
# event_study_r.R - R parity anchor for event_study() using fixest::feols HC2.
#
# Model 1: y ~ post (treated only), HC2 SEs
# Model 2: y ~ post + x (treated only), HC2 SEs
# Matches Stata: regress y post [if treated==1], vce(hc2)
#
# Args: argv[1] = input csv, argv[2] = output json

library(fixest)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
df <- read.csv(args[1])
df_treated <- df[df$treated == 1, ]

# Model 1: no covariates
m1 <- feols(y ~ post, data = df_treated, vcov = "HC2")
se1 <- sqrt(diag(vcov(m1)))
t1 <- coef(m1) / se1
df_r1 <- nobs(m1) - length(coef(m1))
p1 <- 2 * pt(-abs(t1), df = df_r1)
ci_l1 <- coef(m1) - qt(0.975, df = df_r1) * se1
ci_u1 <- coef(m1) + qt(0.975, df = df_r1) * se1

# Model 2: with covariate x
m2 <- feols(y ~ post + x, data = df_treated, vcov = "HC2")
se2 <- sqrt(diag(vcov(m2)))
t2 <- coef(m2) / se2
df_r2 <- nobs(m2) - length(coef(m2))
p2 <- 2 * pt(-abs(t2), df = df_r2)
ci_l2 <- coef(m2) - qt(0.975, df = df_r2) * se2
ci_u2 <- coef(m2) + qt(0.975, df = df_r2) * se2

# Flatten to name-value pairs matching Stata fixture format
# Use unname() to strip named-vector attributes
nms1 <- names(coef(m1))
nms2 <- names(coef(m2))

out <- c(
  list(m1_N = nobs(m1), m1_df_r = df_r1, m1_r2 = unname(r2(m1)[1])),
  as.list(setNames(unname(coef(m1)), paste0("m1_coef_", nms1))),
  as.list(setNames(unname(se1), paste0("m1_se_", nms1))),
  as.list(setNames(unname(t1), paste0("m1_t_", nms1))),
  as.list(setNames(unname(p1), paste0("m1_p_", nms1))),
  as.list(setNames(unname(ci_l1), paste0("m1_ci95l_", nms1))),
  as.list(setNames(unname(ci_u1), paste0("m1_ci95u_", nms1))),
  list(m2_N = nobs(m2), m2_df_r = df_r2, m2_r2 = unname(r2(m2)[1])),
  as.list(setNames(unname(coef(m2)), paste0("m2_coef_", nms2))),
  as.list(setNames(unname(se2), paste0("m2_se_", nms2))),
  as.list(setNames(unname(t2), paste0("m2_t_", nms2))),
  as.list(setNames(unname(p2), paste0("m2_p_", nms2))),
  as.list(setNames(unname(ci_l2), paste0("m2_ci95l_", nms2))),
  as.list(setNames(unname(ci_u2), paste0("m2_ci95u_", nms2)))
)

write_json(out, args[2], auto_unbox = TRUE, digits = 17)
