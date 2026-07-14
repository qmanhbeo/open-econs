#!/usr/bin/env Rscript
# Diagnostic 4: check DOF details from fixest

library(fixest)

df <- read.csv("tests/r/fixtures/inputs/did_sun_abraham_input.csv")

est <- feols(y ~ x + sunab(cohort, time) | entity + time,
             data = df, cluster = ~entity)

cat("=== Model DOF details ===\n")
cat("nobs:", est$nobs, "\n")
cat("nparams:", est$nparams, "\n")
cat("df.residual:", est$df.residual, "\n")
cat("df.model:", est$df.model, "\n")
cat("sigma2:", est$sigma2, "\n")
cat("ssr:", est$ssr, "\n")
cat("r.squared:", est$r.squared, "\n")

cat("\n=== Number of unique values ===\n")
cat("n_entity:", length(unique(df$entity)), "\n")
cat("n_time:", length(unique(df$time)), "\n")
cat("n_cohort:", length(unique(df$cohort[!is.na(df$cohort)])), "\n")

cat("\n=== Summary table ===\n")
ct <- summary(est)$coeftable
print(ct)
cat("\nnrow(coeftable):", nrow(ct), "\n")

# Check vcov with different options
cat("\n=== VCE iid ===\n")
V_iid <- vcov(est, vcov = "iid")
cat("dim:", dim(V_iid), "\n")

cat("\n=== VCE clustered (default) ===\n")
V_cl <- vcov(est)
cat("dim:", dim(V_cl), "\n")
cat("colnames:", colnames(V_cl), "\n")

# Check the nclusters attribute
cat("\n=== est$cluster ===\n")
if (!is.null(est$cluster)) {
  cat("cluster:", est$cluster, "\n")
} else {
  cat("NULL\n")
}

# Check the model's internal info about collinearity
cat("\n=== Internal DOF computation ===\n")
cat("n - nparams:", est$nobs - est$nparams, "\n")

# The number of absorbed parameters
cat("\n=== Absorbed DOF ===\n")
cat("n_entity groups:", length(unique(df$entity)), "\n")
cat("n_time groups:", length(unique(df$time)), "\n")
# In fixest: absorbed DOF = n_groups_entity + n_groups_time - 1 (intersection counted twice)
absorbed <- length(unique(df$entity)) + length(unique(df$time)) - 1
cat("absorbed DOF (inclusion-exclusion):", absorbed, "\n")
cat("n - absorbed - k:", est$nobs - absorbed - length(coef(est, agg=FALSE)), "\n")
cat("est$df.residual:", est$df.residual, "\n")
