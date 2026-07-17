library(plm)
df <- read.csv("tests/r/fixtures/inputs/abond_input.csv")
cat("cols:", paste(names(df), collapse=","), "\n")
df$entity <- as.factor(df$entity)
df$time <- as.integer(df$time)
cat("n:", nrow(df), "entities:", nlevels(df$entity), "\n")
p <- pdata.frame(df, index = c("entity", "time"))
# modern plm: two-part formula; RHS2 = GMM instruments (lags 2..4 of y),
# standard instruments = x, z (lag 0, i.e. current differenced values).
f <- y | lag(y, -1) + lag(x, 0) + lag(z, 0) ~ lag(y, -2:-4) + lag(x, 0) + lag(z, 0)
cat("trying pgmm...\n")
m <- tryCatch(
  pgmm(f, data = p, effect = "twoways", model = "onestep", transformation = "d"),
  error = function(e) { cat("ERR:", conditionMessage(e), "\n"); NULL })
if (!is.null(m)) {
  s <- summary(m, robust = FALSE)
  cat("coefficients:\n"); print(round(coef(m), 6))
  cat("se:\n"); print(round(sqrt(diag(vcov(m, robust = FALSE))), 6))
  cat("wald J stat/pval:", s$wald.stat, s$wald.pval, "\n")
} else {
  cat("pgmm returned NULL\n")
}
