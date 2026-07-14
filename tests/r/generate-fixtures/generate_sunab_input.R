#!/usr/bin/env Rscript
# Generate multi-cohort input data for Sun-Abraham parity testing
# Uses the same base data as did_cs_balanced_input.csv
# but adds a cohort column for staggered treatment

df <- read.csv("tests/r/fixtures/inputs/did_cs_balanced_input.csv")

# Multi-cohort treatment:
# Entities 0-4: never-treated (cohort = NA)
# Entities 5-9: treated at time 2 (cohort = 2)
# Entities 10-14: treated at time 3 (cohort = 3)
# Entities 15-19: treated at time 4 (cohort = 4)
# Entities 20-24: never-treated (cohort = NA)
# Entities 25-29: never-treated (cohort = NA)
df$cohort <- NA_real_
df$cohort[df$entity >= 5 & df$entity < 10] <- 2
df$cohort[df$entity >= 10 & df$entity < 15] <- 3
df$cohort[df$entity >= 15 & df$entity < 20] <- 4

# Write output
write.csv(df, "tests/r/fixtures/inputs/did_sa_input.csv", row.names = FALSE)
cat("Input CSV written to tests/r/fixtures/inputs/did_sa_input.csv\n")
cat("N rows:", nrow(df), "\n")
cat("Cohort distribution:\n")
print(table(df$cohort, useNA = "always"))
