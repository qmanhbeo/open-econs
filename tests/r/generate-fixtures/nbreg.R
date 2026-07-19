#!/usr/bin/env Rscript
# Negative binomial regression parity fixture for open-econs.
# Ground truth = R:
#   - fixest::fenegbin  (FE NB2; fenegbin is NB2-only in fixest 0.14.2)
#   - MASS::glm.nb      (pooled NB2; theta = 1/alpha, NB2 size parameter)
#
# argv[1] = input csv  (tests/r/fixtures/inputs/nbreg_input.csv)
# argv[2] = output json (tests/r/fixtures/expected/nbreg.json)
#
# Records:
#   FE NB2 (fenegbin):  b_x1, b_x2, se_x1, se_x2, theta, loglik
#   pooled NB2 (glm.nb): b_x1_p, b_x2_p, se_x1_p, se_x2_p, theta_p, loglik_p

library(fixest)
library(MASS)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
in_csv  <- args[1]
out_json <- args[2]

df <- read.csv(in_csv)

# --- FE NB2 (fenegbin) ---
m_fe <- fenegbin(y ~ x1 + x2 | firm + year, data = df)
b_x1  <- unname(coef(m_fe)["x1"])
b_x2  <- unname(coef(m_fe)["x2"])
se_x1 <- as.numeric(se(m_fe)["x1"])
se_x2 <- as.numeric(se(m_fe)["x2"])
theta <- as.numeric(m_fe$theta)
loglik <- as.numeric(logLik(m_fe))

# --- pooled NB2 (glm.nb) ---
m_p <- glm.nb(y ~ x1 + x2, data = df)
b_x1_p  <- unname(coef(m_p)["x1"])
b_x2_p  <- unname(coef(m_p)["x2"])
se_x1_p <- summary(m_p)$coefficients["x1", "Std. Error"]
se_x2_p <- summary(m_p)$coefficients["x2", "Std. Error"]
theta_p <- m_p$theta
loglik_p <- as.numeric(logLik(m_p))

out <- list(
  b_x1 = b_x1, b_x2 = b_x2, se_x1 = se_x1, se_x2 = se_x2,
  theta = theta, loglik = loglik,
  b_x1_p = b_x1_p, b_x2_p = b_x2_p, se_x1_p = se_x1_p, se_x2_p = se_x2_p,
  theta_p = theta_p, loglik_p = loglik_p
)

cat("writing", out_json, "\n")
write_json(out, out_json, digits = 15, auto_unbox = TRUE, pretty = TRUE)
