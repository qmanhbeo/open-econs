#!/usr/bin/env Rscript
# Diagnostic 8: verify n_clusters=15 and R² computation

library(fixest)

df <- read.csv("tests/r/fixtures/inputs/did_sun_abraham_input.csv")

cat("=== Treated entities ===\n")
treated_entities <- unique(df$entity[!is.na(df$cohort)])
cat("n treated entities:", length(treated_entities), "\n")
cat("entities:", sort(treated_entities), "\n")

est <- feols(y ~ x + sunab(cohort, time) | entity + time,
             data = df, cluster = ~entity)

cat("\n=== nclusters from fixest internals ===\n")
# Try to access internal nclusters
tryCatch({
  cat("est$llik:", tryCatch(est$llik, error=function(e) "NOT FOUND"), "\n")
}, error=function(e) NULL)

# Check what nobs fixest reports
cat("\nest$nobs:", est$nobs, "\n")

# Check R² computation from the fixture script formula
r_squared <- 1 - est$sigma2 * (est$nobs - est$nparams) /
  (sum((df$y - mean(df$y))^2, na.rm = TRUE) / (est$nobs - 1))
cat("r_squared (fixture formula):", r_squared, "\n")

# Check alternative R² formulas
cat("\n=== Alternative R² computations ===\n")
# Standard within R²
y_within <- est$residuals + est$fitted.values
ssr <- sum(est$residuals^2)
cat("SSR:", ssr, "\n")

# SST using all observations
sst_all <- sum((df$y - mean(df$y))^2, na.rm=TRUE)
cat("SST (all obs):", sst_all, "\n")

# SST using only estimation sample
df_used <- df[!is.na(df$cohort), ]
sst_used <- sum((df_used$y - mean(df_used$y))^2)
cat("SST (used obs):", sst_used, "\n")

# R² = 1 - SSR/SST (standard, using estimation sample)
cat("R² (standard, used obs):", 1 - ssr/sst_used, "\n")

# R² = 1 - SSR/SST (all obs)
cat("R² (all obs):", 1 - ssr/sst_all, "\n")

# R² = 1 - (SSR/(n-k)) / (SST/(n-1)) (fixture formula)
# est$sigma2 = SSR/(n-k)
cat("sigma2:", est$sigma2, "\n")
cat("n-nparams:", est$nobs - est$nparams, "\n")
cat("SSR check:", est$sigma2 * (est$nobs - est$nparams), "\n")

# The fixture formula:
# r² = 1 - sigma2 * (n - nparams) / (SST_all / (n - 1))
# = 1 - SSR / (SST_all / 74)
# = 1 - SSR * 74 / SST_all
cat("R² (fixture formula):", 1 - ssr * (est$nobs - 1) / sst_all, "\n")
