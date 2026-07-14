# Multinomial logit parity vs R nnet::multinom (coefficients only).
# Reads committed input CSV (argv[1]); writes expected-output JSON (argv[2]).
# Output shape: {"coef": {"<category>": {"(Intercept)":v, "x1":v, "x2":v}}}.
# Mirrors TestMlogitR.test_r_coefficients in tests/stata/test_mlogit.py.
# The baseline category is pinned to 1 (factor levels 1:3) so it matches
# open-econs base=1 and Stata baseoutcome(1).
library(nnet)
library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
csv <- args[1]
out_json <- args[2]
df <- read.csv(csv)
df$y <- factor(df$y, levels = c(1, 2, 3))
fit <- multinom(y ~ x1 + x2, data = df, trace = FALSE, maxit = 500)
cm <- as.matrix(coef(fit))
out <- list(coef = list())
for (cat in rownames(cm)) {
  out$coef[[cat]] <- list(
    "(Intercept)" = as.numeric(cm[cat, "(Intercept)"]),
    "x1" = as.numeric(cm[cat, "x1"]),
    "x2" = as.numeric(cm[cat, "x2"])
  )
}
write_json(out, out_json, auto_unbox = TRUE, digits = 15)
