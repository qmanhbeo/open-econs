#!/usr/bin/env Rscript
# Diagnostic 7: check df for ALL t-tests in the model

library(fixest)

df <- read.csv("tests/r/fixtures/inputs/did_sun_abraham_input.csv")

est <- feols(y ~ x + sunab(cohort, time) | entity + time,
             data = df, cluster = ~entity)

# Get the summary coeftable
ct <- summary(est)$coeftable
cat("=== Summary coeftable ===\n")
print(ct)

# For each coefficient, find which df gives the matching p-value
cat("\n=== df for each coefficient t-test ===\n")
for (i in 1:nrow(ct)) {
  t_val <- ct[i, "t value"]
  p_val <- ct[i, "Pr(>|t|)"]
  for (df_val in 1:74) {
    p_calc <- 2 * pt(-abs(t_val), df=df_val)
    if (abs(p_calc - p_val) < 1e-10) {
      cat(sprintf("  %s: t=%.6f, df=%d, p=%.6e\n", rownames(ct)[i], t_val, df_val, p_val))
      break
    }
  }
}

# ATT aggregation
att_agg <- aggregate(est, agg = "att")
cat("\n=== ATT aggregation ===\n")
print(att_agg)

# ATT df
att_t <- unname(att_agg[1, "t value"])
att_p <- unname(att_agg[1, "Pr(>|t|)"])
for (df_val in 1:74) {
  p_calc <- 2 * pt(-abs(att_t), df=df_val)
  if (abs(p_calc - att_p) < 1e-10) {
    cat(sprintf("  ATT: t=%.6f, df=%d, p=%.6e\n", att_t, df_val, att_p))
    break
  }
}

# Period aggregation
period_agg <- aggregate(est, agg = "period")
cat("\n=== Period aggregation ===\n")
print(period_agg)

# For each period aggregate, find df
cat("\n=== df for each period aggregate ===\n")
for (i in 1:nrow(period_agg)) {
  t_val <- period_agg[i, "t value"]
  p_val <- period_agg[i, "Pr(>|t|)"]
  for (df_val in 1:74) {
    p_calc <- 2 * pt(-abs(t_val), df=df_val)
    if (abs(p_calc - p_val) < 1e-10) {
      cat(sprintf("  %s: t=%.6f, df=%d, p=%.6e\n", rownames(period_agg)[i], t_val, df_val, p_val))
      break
    }
  }
}

# Check nclusters
cat("\n=== Number of clusters used ===\n")
cat("n clusters (entity):", length(unique(df$entity)), "\n")

# Check what fixest uses internally for the VCE df
cat("\n=== fixest vcov internals ===\n")
cat("vcov type:", est$vocov_type, "\n")
cat("cluster:", est$cluster, "\n")
cat("dof.K:", tryCatch(est$dof.K, error=function(e) "NOT FOUND"), "\n")
cat("dof.G:", tryCatch(est$dof.G, error=function(e) "NOT FOUND"), "\n")
cat("dof.T:", tryCatch(est$dof.T, error=function(e) "NOT FOUND"), "\n")
