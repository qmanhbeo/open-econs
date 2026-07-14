#!/usr/bin/env Rscript
# Diagnostic: understand what coef(est) vs vcov(est) return for sunab model

library(fixest)

df <- read.csv("tests/r/fixtures/inputs/did_sun_abraham_input.csv")

cat("=== Data structure ===\n")
cat("N rows:", nrow(df), "\n")
cat("Unique entities:", length(unique(df$entity)), "\n")
cat("Unique times:", length(unique(df$time)), "\n")
cat("Cohort values:", sort(unique(df$cohort[!is.na(df$cohort)])), "\n")
cat("Never-treated:", sum(is.na(df$cohort)), "obs\n")
cat("Treated:", sum(!is.na(df$cohort)), "obs\n\n")

est <- feols(y ~ x + sunab(cohort, time) | entity + time,
             data = df, cluster = ~entity)

cat("=== coef(est) ===\n")
c <- coef(est)
cat("length:", length(c), "\n")
cat("names:", names(c), "\n")
cat("values:", c, "\n\n")

cat("=== vcov(est) ===\n")
V <- vcov(est)
cat("dim:", dim(V), "\n")
cat("rownames:", rownames(V), "\n")
cat("colnames:", colnames(V), "\n\n")

cat("=== coef(est) length vs vcov dim ===\n")
cat("coef length:", length(c), "\n")
cat("vcov nrow:", nrow(V), "\n")
cat("vcov ncol:", ncol(V), "\n")
cat("MATCH:", length(c) == nrow(V), "\n\n")

# Check if coef has an agg argument
cat("=== coef.fixest args ===\n")
cat(args(coef.fixest), "\n\n")

# Try coef with different agg settings
cat("=== coef(est, agg=FALSE) ===\n")
c_full <- coef(est, agg = FALSE)
cat("length:", length(c_full), "\n")
cat("names:", names(c_full), "\n")
cat("values:", c_full, "\n\n")

# Check collin.var
cat("=== collin.var ===\n")
cat("collin.var:", est$collin.var, "\n\n")

# Check model.matrix
cat("=== model.matrix(est) ===\n")
mm <- model.matrix(est)
cat("dim:", dim(mm), "\n")
cat("colnames:", colnames(mm), "\n\n")

# Check model_matrix_info
cat("=== model_matrix_info ===\n")
cat("names:", names(est$model_matrix_info), "\n")
if (!is.null(est$model_matrix_info$sunab)) {
  cat("sunab info:\n")
  str(est$model_matrix_info$sunab)
}

# Summary
cat("\n=== summary(est)$coeftable ===\n")
ct <- summary(est)$coeftable
print(ct)
cat("nrow(coeftable):", nrow(ct), "\n")
