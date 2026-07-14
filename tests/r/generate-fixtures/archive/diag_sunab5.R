#!/usr/bin/env Rscript
# Diagnostic 5: check exact df used for t-tests in fixest

library(fixest)

df <- read.csv("tests/r/fixtures/inputs/did_sun_abraham_input.csv")

est <- feols(y ~ x + sunab(cohort, time) | entity + time,
             data = df, cluster = ~entity)

cat("=== Key model info ===\n")
cat("nobs:", est$nobs, "\n")
cat("nparams:", est$nparams, "\n")

# Try to access internal DOF info
cat("\n=== Searching for df info ===\n")
cat("df.residual:", tryCatch(est$df.residual, error=function(e) "NOT FOUND"), "\n")
cat("df.tss:", tryCatch(est$df.tss, error=function(e) "NOT FOUND"), "\n")
cat("df.null:", tryCatch(est$df.null, error=function(e) "NOT FOUND"), "\n")

# Check what summary uses for df
cat("\n=== Manual t-test computation ===\n")
ct <- summary(est)$coeftable
for (i in 1:nrow(ct)) {
  coef <- ct[i, "Estimate"]
  se <- ct[i, "Std. Error"]
  t <- ct[i, "t value"]
  p <- ct[i, "Pr(>|t|)"]
  # Try different df values
  for (df_val in c(28, 29, 47, 74)) {
    p_calc <- 2 * pt(-abs(t), df=df_val)
    if (abs(p_calc - p) < 1e-6) {
      cat(sprintf("  %s: t=%.4f, p=%.6e, df=%d matches\n", rownames(ct)[i], t, p, df_val))
      break
    }
  }
}

# Check the number of absorbed parameters more carefully
cat("\n=== Absorbed parameter count ===\n")
# In fixest, nparams includes both estimated and absorbed params
cat("nparams:", est$nparams, "\n")
cat("n - nparams:", est$nobs - est$nparams, "\n")
# The residual df should be n - nparams
cat("Expected residual df:", est$nobs - est$nparams, "\n")

# Check cluster info
cat("\n=== Cluster info ===\n")
cat("n clusters:", length(unique(df$entity)), "\n")
cat("G - 1:", length(unique(df$entity)) - 1, "\n")
