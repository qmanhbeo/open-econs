library(plm)

args <- commandArgs(trailingOnly=TRUE)
in_csv <- args[1]
out_json <- args[2]

df <- read.csv(in_csv)
# entity/time already numeric/int
df$entity <- as.factor(df$entity)
df$time <- as.integer(df$time)
p <- pdata.frame(df, index = c("entity", "time"))

# Stata fixture for comparison
stata <- NULL
tryCatch({
  st <- haven::read_dta("tests/stata/fixtures/expected/abond.dta")
  stata <- as.list(st)
}, error = function(e) { cat("no stata dta:", conditionMessage(e), "\n") })

pgmm_fit <- function(step, robust, collapse) {
  # pgmm cannot "collapse"; collapse reduces GMM instruments to lag(2) only.
  # Build formula: y ~ lag(y,-1) + lag(x,0) + lag(z,0)
  # GMM instruments on L.y: in pgmm, ~ lag(y, -1:-(maxlag))
  f <- y ~ lag(y, -1) + lag(x, 0) + lag(z, 0)
  # pgmm gmm.inst controls which vars get GMM instruments.
  # Default pgmm: GMM on all lagged dependent vars with lags -2:-(t-1) effectively.
  # To match xtabond2 gmm(L.y, lag(2 4)) we restrict, but pgmm does not allow
  # arbitrary lag windows easily. Compare with default pgmm first.
  m <- tryCatch(
    pgmm(f, data = p,
         effect = "twoways",  # difference GMM uses "individual"? Actually twoways for fd.
         model = "onestep",
         transformation = "d",
         collapse = collapse),
    error = function(e) NULL
  )
  if (is.null(m)) return(NULL)
  s <- summary(m, robust = robust)
  list(coef = coef(m), se = sqrt(diag(vcov(m, robust = robust))),
       J = s$wald.stat, pJ = s$wald.pval)
}

for (step in c("onestep","twosteps")) {
  for (robust in c(FALSE, TRUE)) {
    for (collapse in c(FALSE, TRUE)) {
      r <- pgmm_fit(step, robust, collapse)
      cat("=== step:", step, "robust:", robust, "collapse:", collapse, "===\n")
      if (is.null(r)) { cat("  NULL\n"); next }
      print(round(r$coef, 6))
      print(round(r$se, 6))
    }
  }
}
cat("DONE\n")
