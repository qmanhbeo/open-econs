#!/usr/bin/env Rscript
# Quick exploration of fixest::sunab() to understand estimator structure

library(fixest)
library(jsonlite)

df <- read.csv("tests/r/fixtures/inputs/staggered_did_balanced_input.csv")

# Staggered treatment: entities 10-19 treated at time >= 3
df$cohort <- ifelse(df$entity >= 10 & df$entity < 20, 3, NA)

cat("=== Test 1: sunab without time FE ===\n")
est1 <- feols(y ~ x + sunab(cohort, time) | entity, data = df, cluster = ~entity)
print(summary(est1))
cat("Coefficients:", names(coef(est1)), "\n")
cat("Values:", coef(est1), "\n\n")

cat("=== Test 2: sunab with time FE ===\n")
est2 <- feols(y ~ x + sunab(cohort, time) | entity + time, data = df, cluster = ~entity)
print(summary(est2))
cat("Coefficients:", names(coef(est2)), "\n")
cat("Values:", coef(est2), "\n\n")

cat("=== Test 3: sunab_att without time FE ===\n")
est3 <- feols(y ~ x + sunab_att(cohort, time) | entity, data = df, cluster = ~entity)
print(summary(est3))
cat("Coefficients:", names(coef(est3)), "\n")
cat("Values:", coef(est3), "\n\n")

cat("=== Test 4: sunab with ref.p=-1, ref.c=NULL (never-treated) ===\n")
est4 <- feols(y ~ x + sunab(cohort, time, ref.p = -1) | entity, data = df, cluster = ~entity)
print(summary(est4))
cat("Coefficients:", names(coef(est4)), "\n")
cat("Values:", coef(est4), "\n\n")

# Check what sunab() returns as a matrix
cat("=== sunab matrix (first 10 rows) ===\n")
sunab_mat <- sunab(df$cohort, df$time)
cat("Dimensions:", dim(sunab_mat), "\n")
cat("Column names:", colnames(sunab_mat), "\n")
print(head(sunab_mat, 10))

# Check the model matrix
cat("\n=== Model matrix for est1 (first 10 rows) ===\n")
mm1 <- model.matrix(est1)
cat("Dimensions:", dim(mm1), "\n")
cat("Column names:", colnames(mm1), "\n")
print(head(mm1, 10))

cat("\n=== Model matrix for est3 (first 10 rows) ===\n")
mm3 <- model.matrix(est3)
cat("Dimensions:", dim(mm3), "\n")
cat("Column names:", colnames(mm3), "\n")
print(head(mm3, 10))
