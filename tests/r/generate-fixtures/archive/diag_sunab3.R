#!/usr/bin/env Rscript
# Diagnostic 3: get raw coefficient vector (unaggregated)

library(fixest)

df <- read.csv("tests/r/fixtures/inputs/did_sun_abraham_input.csv")

est <- feols(y ~ x + sunab(cohort, time) | entity + time,
             data = df, cluster = ~entity)

# Try different coef arguments
cat("=== coef(est, agg=FALSE) ===\n")
tryCatch({
  c_full <- coef(est, agg = FALSE)
  cat("length:", length(c_full), "\n")
  cat("names:", names(c_full), "\n")
  cat("values:", c_full, "\n")
}, error = function(e) cat("ERROR:", e$message, "\n"))

cat("\n=== coef(est, collin.rm=FALSE) ===\n")
tryCatch({
  c_nocollin <- coef(est, collin.rm = FALSE)
  cat("length:", length(c_nocollin), "\n")
  cat("names:", names(c_nocollin), "\n")
  cat("values:", c_nocollin, "\n")
}, error = function(e) cat("ERROR:", e$message, "\n"))

# Check fixest internal structure for raw coefficients
cat("\n=== est$coefficients ===\n")
if (!is.null(est$coefficients)) {
  cat("length:", length(est$coefficients), "\n")
  cat("names:", names(est$coefficients), "\n")
  cat("values:", est$coefficients, "\n")
} else {
  cat("NULL\n")
}

cat("\n=== est$coefficients_df ===\n")
if (!is.null(est$coefficients_df)) {
  print(est$coefficients_df)
} else {
  cat("NULL\n")
}

# Check if we can access the internal collin info to reconstruct
cat("\n=== est$collin.coef ===\n")
if (!is.null(est$collin.coef)) {
  cat("length:", length(est$collin.coef), "\n")
  cat("names:", names(est$collin.coef), "\n")
  cat("values:", est$collin.coef, "\n")
} else {
  cat("NULL\n")
}

cat("\n=== est$collin.var ===\n")
cat(est$collin.var, "\n")

# Check summary coeftable
cat("\n=== summary(est)$coeftable ===\n")
ct <- summary(est)$coeftable
print(ct)
cat("nrow:", nrow(ct), "\n")

# Check fixest version
cat("\n=== fixest version ===\n")
print(packageVersion("fixest"))

# Check args of coef method
cat("\n=== methods(coef) ===\n")
print(methods(coef))

# Try to look at the coef.fixest source
cat("\n=== coef.fixest source ===\n")
print(getS3method("coef", "fixest"))
