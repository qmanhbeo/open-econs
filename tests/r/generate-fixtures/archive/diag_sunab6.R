#!/usr/bin/env Rscript
# Diagnostic 6: find exact df used for ATT t-test

library(fixest)

df <- read.csv("tests/r/fixtures/inputs/did_sun_abraham_input.csv")

est <- feols(y ~ x + sunab(cohort, time) | entity + time,
             data = df, cluster = ~entity)

att_agg <- aggregate(est, agg = "att")
att_coef <- unname(att_agg[1, "Estimate"])
att_se <- unname(att_agg[1, "Std. Error"])
att_t <- unname(att_agg[1, "t value"])
att_p <- unname(att_agg[1, "Pr(>|t|)"])

cat("ATT coef:", att_coef, "\n")
cat("ATT SE:", att_se, "\n")
cat("ATT t:", att_t, "\n")
cat("ATT p:", att_p, "\n\n")

# Search for which df gives this p-value
cat("=== Searching for matching df ===\n")
for (df_val in 1:74) {
  p_calc <- 2 * pt(-abs(att_t), df=df_val)
  if (abs(p_calc - att_p) < 1e-10) {
    cat(sprintf("MATCH: df=%d gives p=%.15e (target: %.15e)\n", df_val, p_calc, att_p))
  }
}

# Also check the summary table t-tests
cat("\n=== Summary table t-test df search ===\n")
ct <- summary(est)$coeftable
for (i in 1:nrow(ct)) {
  t <- ct[i, "t value"]
  p <- ct[i, "Pr(>|t|)"]
  for (df_val in 25:50) {
    p_calc <- 2 * pt(-abs(t), df=df_val)
    if (abs(p_calc - p) < 1e-10) {
      cat(sprintf("  %s: t=%.6f, p=%.6e, df=%d\n", rownames(ct)[i], t, p, df_val))
      break
    }
  }
}

# Check fixest's internal vcov processing
cat("\n=== fixest version ===\n")
print(packageVersion("fixest"))

# Check what the summary does with df
cat("\n=== est$vocov_details ===\n")
print(est$vocov_details)
