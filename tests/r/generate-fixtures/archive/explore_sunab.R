#!/usr/bin/env Rscript
# Quick exploration of fixest::sunab() to understand estimator structure

library(fixest)
library(jsonlite)

df <- read.csv("tests/r/fixtures/inputs/staggered_did_balanced_input.csv")

# Staggered treatment: entities 10-19 treated at time >= 3
# Need a cohort column: when each unit is first treated
# Entities 0-9: never-treated (cohort = NA)
# Entities 10-19: treated at time 3 (cohort = 3)
# Entities 20-29: never-treated (cohort = NA)
df$cohort <- ifelse(df$entity >= 10 & df$entity < 20, 3, NA)

cat("=== Data structure ===\n")
cat("N rows:", nrow(df), "\n")
cat("Unique entities:", length(unique(df$entity)), "\n")
cat("Unique times:", length(unique(df$time)), "\n")
cat("Cohort values:", sort(unique(df$cohort[!is.na(df$cohort)])), "\n")
cat("Never-treated:", sum(is.na(df$cohort)), "obs\n")
cat("Treated:", sum(!is.na(df$cohort)), "obs\n\n")

# Sun-Abraham estimation
# Formula: y ~ x + sunab(cohort, time) | entity + time
# This creates period x cohort interactions, drops reference period (-1) and never-treated
est <- feols(y ~ x + sunab(cohort, time) | entity + time, data = df, cluster = ~entity)

cat("=== feols summary ===\n")
print(summary(est))

cat("\n=== Coefficients ===\n")
print(coef(est))

cat("\n=== Coefficient names ===\n")
print(names(coef(est)))

cat("\n=== Model matrix columns (first 5 rows) ===\n")
mm <- model.matrix(est)
cat("Dimensions:", dim(mm), "\n")
cat("Column names (first 20):", head(colnames(mm), 20), "\n\n")

# Aggregation: ATT
cat("=== ATT aggregation (default) ===\n")
att <- aggregate(est, agg = "att")
print(att)

cat("\n=== Period aggregation ===\n")
period_agg <- aggregate(est, agg = "period")
print(period_agg)

cat("\n=== Cohort aggregation ===\n")
cohort_agg <- aggregate(est, agg = "cohort")
print(cohort_agg)

# Full coefficients (no aggregation)
cat("\n=== Full coefficients (no_agg via agg=FALSE) ===\n")
full <- aggregate(est, agg = FALSE)
print(full)

# Check is_sunab flag
cat("\n=== is_sunab flag ===\n")
cat("is_sunab:", est$is_sunab, "\n")

# Check model_matrix_info
cat("\n=== model_matrix_info$sunab ===\n")
print(str(est$model_matrix_info$sunab))

# Export fixture
ct <- summary(est)$coeftable
r_squared <- 1 - est$sigma2 * (est$nobs - est$nparams) / (sum((df$y - mean(df$y))^2) / (est$nobs - 1))

# Get the full VCE
V <- vcov(est)
cat("\n=== VCE dimensions ===\n")
cat("VCE:", dim(V), "\n")
cat("VCE colnames:", head(colnames(V), 10), "\n\n")

# Get ATT variance from the aggregate
att_agg <- aggregate(est, agg = "att")
cat("=== ATT SE from aggregate ===\n")
cat("ATT:", att_agg[1, "Estimate"], "\n")
cat("SE:", att_agg[1, "Std. Error"], "\n")
cat("t-value:", att_agg[1, "t value"], "\n")
cat("p-value:", att_agg[1, "Pr(>|t|)"], "\n")

# Export the fixture
out <- list(
  N = est$nobs,
  att = unname(att_agg[1, "Estimate"]),
  se = unname(att_agg[1, "Std. Error"]),
  t_stat = unname(att_agg[1, "t value"]),
  p_value = unname(att_agg[1, "Pr(>|t|)"]),
  r_squared = r_squared,
  sigma2 = est$sigma2,
  # VCE of the full model
  vce_iid = as.list(c(vcov(est, vcov = "iid"))),
  vce_clustered = as.list(c(vcov(est))),
  # All coefficients (full event study)
  coef_names = names(coef(est)),
  coefficients = as.list(unname(coef(est))),
  # ATT aggregation details
  att_coef = unname(att_agg[1, "Estimate"]),
  att_se = unname(att_agg[1, "Std. Error"]),
  # Period aggregation
  period_coefs = as.list(unname(period_agg[, "Estimate"])),
  period_ses = as.list(unname(period_agg[, "Std. Error"])),
  period_names = rownames(period_agg)
)

write_json(out, "tests/r/fixtures/expected/did_sun_abraham.json", auto_unbox = TRUE, digits = 17)
cat("\nFixture written to tests/r/fixtures/expected/did_sun_abraham.json\n")
