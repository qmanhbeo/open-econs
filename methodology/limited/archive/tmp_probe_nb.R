library(fixest)
library(jsonlite)

df <- read.csv("tests/r/fixtures/inputs/nbreg_input.csv")

# fenegbin default (NB2)
m2 <- fenegbin(y ~ x1 + x2 | firm + year, data = df)
cat("=== fenegbin NB2 (default) ===\n")
print(summary(m2))
cat("coef x1:", unname(coef(m2)["x1"]), "\n")
cat("coef x2:", unname(coef(m2)["x2"]), "\n")
# dispersion param: fenegbin reports phi (NB2) or theta?
cat("theta:", as.numeric(m2$theta), "\n")  # fixest fenegbin stores theta
cat("phi:", as.numeric(m2$phi), "\n")
cat("over_disp:", as.numeric(m2$over_disp), "\n")
cat("loglik:", as.numeric(logLik(m2)), "\n")

# fenegbin NB1
m1 <- fenegbin(y ~ x1 + x2 | firm + year, data = df, dispersion = "NB1")
cat("=== fenegbin NB1 ===\n")
cat("coef x1:", unname(coef(m1)["x1"]), "\n")
cat("theta:", as.numeric(m1$theta), "\n")
cat("phi:", as.numeric(m1$phi), "\n")
cat("over_disp:", as.numeric(m1$over_disp), "\n")
