#!/usr/bin/env Rscript
# Diagnostic 2: get raw coefficient vector matching VCE dimensions

library(fixest)

df <- read.csv("tests/r/fixtures/inputs/did_sun_abraham_input.csv")

est <- feols(y ~ x + sunab(cohort, time) | entity + time,
             data = df, cluster = ~entity)

cat("=== coef(est) default ===\n")
c_default <- coef(est)
cat("length:", length(c_default), "\n")
cat("names:", names(c_default), "\n\n")

# Try to get the raw model matrix column names (the full parameter space)
cat("=== model.matrix(est) columns ===\n")
mm <- model.matrix(est)
cat("dim:", dim(mm), "\n")
cat("colnames:", colnames(mm), "\n\n")

# The vcov rownames ARE the full parameter space
V <- vcov(est)
cat("=== vcov rownames ===\n")
cat(rownames(V), "\n\n")

# Check if the model has internal info about the full parameter space
cat("=== est$collin.var ===\n")
cat(est$collin.var, "\n\n")

cat("=== est$model_matrix_info ===\n")
cat("names:", names(est$model_matrix_info), "\n")
for (nm in names(est$model_matrix_info)) {
  cat(sprintf("  %s: %s\n", nm, class(est$model_matrix_info[[nm]])))
}

# Check if sunab creates a specific structure
cat("\n=== est$model_matrix_info$sunab ===\n")
str(est$model_matrix_info$sunab)

# The key question: can we reconstruct the full 9-element coefficient vector?
# The collinear variables were dropped. In the raw model, there were 9 params:
# x + 8 interaction dummies. After dropping 4 collinear, we have 5.
# The VCE is 9x9 (full parameter space).
# We need the 9 raw coefficients, including the 4 collinear ones.

# One approach: re-fit without time FE (which creates collinearity)
cat("\n=== Re-fit without time FE ===\n")
est_notime <- feols(y ~ x + sunab(cohort, time) | entity,
                    data = df, cluster = ~entity)
cat("coef names (no time FE):", names(coef(est_notime)), "\n")
cat("coef length:", length(coef(est_notime)), "\n")
V_notime <- vcov(est_notime)
cat("vcov dim:", dim(V_notime), "\n")
cat("MATCH:", length(coef(est_notime)) == nrow(V_notime), "\n\n")

# Now try the full model again - check fixest version
cat("=== fixest version ===\n")
cat(packageVersion("fixest"), "\n")

# Check if coef.fixest has arguments we can use
cat("\n=== coef.fixest formals ===\n")
print(formals(coef.fixest))
