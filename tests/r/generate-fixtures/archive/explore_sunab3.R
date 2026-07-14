#!/usr/bin/env Rscript
# Exploration of fixest::sunab() with multi-cohort data

library(fixest)
library(jsonlite)

df <- read.csv("tests/r/fixtures/inputs/staggered_did_balanced_input.csv")

# Multi-cohort treatment:
# Entities 0-4: never-treated (cohort = NA)
# Entities 5-9: treated at time 2 (cohort = 2)
# Entities 10-14: treated at time 3 (cohort = 3)
# Entities 15-19: treated at time 4 (cohort = 4)
# Entities 20-24: never-treated (cohort = NA)
# Entities 25-29: never-treated (cohort = NA)
df$cohort <- NA_real_
df$cohort[df$entity >= 5 & df$entity < 10] <- 2
df$cohort[df$entity >= 10 & df$entity < 15] <- 3
df$cohort[df$entity >= 15 & df$entity < 20] <- 4

cat("=== Data structure ===\n")
cat("N rows:", nrow(df), "\n")
cat("Unique entities:", length(unique(df$entity)), "\n")
cat("Unique times:", length(unique(df$time)), "\n")
cat("Cohort values:", sort(unique(df$cohort[!is.na(df$cohort)])), "\n")
cat("Never-treated:", sum(is.na(df$cohort)), "obs\n")
cat("Treated:", sum(!is.na(df$cohort)), "obs\n")
cat("Entities per cohort:\n")
for (c in sort(unique(df$cohort[!is.na(df$cohort)]))) {
  cat("  cohort", c, ":", length(unique(df$entity[df$cohort == c])), "entities\n")
}
cat("Never-treated entities:", length(unique(df$entity[is.na(df$cohort)])), "\n\n")

# Test with multiple cohorts
cat("=== Test 1: sunab with time FE, multiple cohorts ===\n")
est1 <- feols(y ~ x + sunab(cohort, time) | entity + time, data = df, cluster = ~entity)
print(summary(est1))
cat("Coefficients:", names(coef(est1)), "\n")
cat("Values:", coef(est1), "\n\n")

cat("=== Test 2: sunab without time FE, multiple cohorts ===\n")
est2 <- feols(y ~ x + sunab(cohort, time) | entity, data = df, cluster = ~entity)
print(summary(est2))
cat("Coefficients:", names(coef(est2)), "\n")
cat("Values:", coef(est2), "\n\n")

cat("=== Test 3: sunab_att, multiple cohorts ===\n")
est3 <- feols(y ~ x + sunab_att(cohort, time) | entity, data = df, cluster = ~entity)
print(summary(est3))
cat("Coefficients:", names(coef(est3)), "\n")
cat("Values:", coef(est3), "\n\n")

# Aggregate ATT
cat("=== ATT aggregation (est1) ===\n")
att1 <- aggregate(est1, agg = "att")
print(att1)

cat("\n=== Period aggregation (est1) ===\n")
period1 <- aggregate(est1, agg = "period")
print(period1)

cat("\n=== Cohort aggregation (est1) ===\n")
cohort1 <- aggregate(est1, agg = "cohort")
print(cohort1)

# Export fixture data
cat("\n=== Exporting fixture ===\n")
att_agg <- aggregate(est1, agg = "att")
period_agg <- aggregate(est1, agg = "period")
cohort_agg <- aggregate(est1, agg = "cohort")

r_squared <- 1 - est1$sigma2 * (est1$nobs - est1$nparams) / (sum((df$y - mean(df$y))^2) / (est1$nobs - 1))

out <- list(
  N = est1$nobs,
  att = unname(att_agg[1, "Estimate"]),
  se = unname(att_agg[1, "Std. Error"]),
  t_stat = unname(att_agg[1, "t value"]),
  p_value = unname(att_agg[1, "Pr(>|t|)"]),
  r_squared = r_squared,
  sigma2 = est1$sigma2,
  # VCE of the full model
  vce_iid = as.list(c(vcov(est1, vcov = "iid"))),
  vce_clustered = as.list(c(vcov(est1))),
  # All coefficients (full event study)
  coef_names = names(coef(est1)),
  coefficients = as.list(unname(coef(est1))),
  # ATT aggregation
  att_coef = unname(att_agg[1, "Estimate"]),
  att_se = unname(att_agg[1, "Std. Error"]),
  # Period aggregation
  period_coefs = as.list(unname(period_agg[, "Estimate"])),
  period_ses = as.list(unname(period_agg[, "Std. Error"])),
  period_names = rownames(period_agg),
  # Cohort aggregation
  cohort_coefs = as.list(unname(cohort_agg[, "Estimate"])),
  cohort_ses = as.list(unname(cohort_agg[, "Std. Error"])),
  cohort_names = rownames(cohort_agg)
)

write_json(out, "tests/r/fixtures/expected/did_sun_abraham.json", auto_unbox = TRUE, digits = 17)
cat("Fixture written to tests/r/fixtures/expected/did_sun_abraham.json\n")
