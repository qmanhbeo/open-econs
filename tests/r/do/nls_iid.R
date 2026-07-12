# NLS parity vs R nls() -- iid (nonrobust) coefficients and SEs.
# Reads committed input CSV (argv[1]); writes expected-output JSON (argv[2]).
# Output shape: {"coef": {a,b,c}, "se": {a,b,c}}.  Mirrors tests/test_nls.py.
library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
csv <- args[1]
out_json <- args[2]
df <- read.csv(csv)
fit <- nls(y ~ a*exp(-b*x)+c, data = df, start = list(a = 1, b = 1, c = 0))
cf <- coef(fit)
se <- sqrt(diag(vcov(fit)))
out <- list(
  coef = list(a = as.numeric(cf["a"]), b = as.numeric(cf["b"]), c = as.numeric(cf["c"])),
  se   = list(a = as.numeric(se["a"]), b = as.numeric(se["b"]), c = as.numeric(se["c"]))
)
write_json(out, out_json, auto_unbox = TRUE, digits = 15)
