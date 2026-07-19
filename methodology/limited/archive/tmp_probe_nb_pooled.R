library(MASS)       # glm.nb (NB2, pooled)
library(jsonlite)

df <- read.csv("tests/r/fixtures/inputs/nbreg_input.csv")

# glm.nb: NB2 pooled, reports theta (size)
gnb <- glm.nb(y ~ x1 + x2, data = df)
cat("=== glm.nb (NB2 pooled) ===\n")
cat("coef x1:", unname(coef(gnb)["x1"]), "\n")
cat("coef x2:", unname(coef(gnb)["x2"]), "\n")
cat("theta (size):", gnb$theta, "\n")
cat("loglik:", as.numeric(logLik(gnb)), "\n")
cat("se x1:", summary(gnb)$coefficients["x1","Std. Error"], "\n")

# countreg::nbreg not available; use AER::glm.nb? just report glm.nb NB2.
cat("countreg not installed; skipping NB1 R pooled probe.\n")
