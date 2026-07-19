#! diag_estat.R -- R ground truth for open_econs OLS post-estimation diagnostics.
#
# Reference for v1.3 diagnostics parity (Breusch-Godfrey, White's general test,
# Ljung-Box, Cook's distance, leverage/hatvalues, DFBETAS).  Fits
#   fit <- lm(y ~ x1 + x2, data = df)
# on the committed df_ols input and serializes every reference quantity.
#
# Conventions captured (source-confirmed 2026-07-19, run against R 4.6.1):
#
#  * Breusch-Godfrey: lmtest::bgtest(fit, order=L).  bgtest's auxiliary
#    regression includes the FULL original design (constant + x1 + x2) plus L
#    lagged residuals, and the LM statistic is n * R^2 ~ chi2(L).  This is the
#    same construction as open_econs' hand-rolled breusch_godfrey (statsmodels'
#    acorr_breusch_godfrey dropped the design matrix, hence OE reimplemented it).
#    Captured for order = 1 and order = 2.
#
#  * White's general test: replicated EXACTLY as open_econs builds it -- regress
#    resid^2 on the non-constant regressors [x1, x2], their squares
#    [x1^2, x2^2], and the pairwise cross-product [x1*x2], each MEAN-CENTERED,
#    with an auxiliary intercept.  LM = n * R^2 ~ chi2(5).  (No off-the-shelf R
#    command produces this identical auxiliary; bptest with a supplied formula
#    differs in centering, so we build the auxiliary by hand to match OE.)
#
#  * Ljung-Box: Box.test(resid(fit), lag=1, type="Ljung-Box").
#
#  * Cook's distance: cooks.distance(fit)  -- full n-vector.
#  * Leverage: hatvalues(fit)              -- full n-vector.
#  * DFBETAS: dfbetas(fit)                 -- full n x 3 matrix, cols
#    (Intercept, x1, x2).  R standardizes by the leave-one-out residual sd
#    sigma_(-i) = sqrt( (SSR - e_i^2/(1-h_i)) / (n-k-1) ) (== lm.influence()$sigma).
#    open_econs uses a different LOO-variance formula (missing the 1/(1-h_i)
#    factor), a documented ~1e-4 divergence -- see test_r_diagnostics.py xfail.
args <- commandArgs(trailingOnly = TRUE)
in_csv   <- args[1]
out_json <- args[2]

suppressMessages(library(lmtest))
suppressMessages(library(jsonlite))

df <- read.csv(in_csv)
fit <- lm(y ~ x1 + x2, data = df)

res <- resid(fit)
n <- length(res)

## --- Breusch-Godfrey (orders 1 and 2) ---
bg1 <- bgtest(fit, order = 1)
bg2 <- bgtest(fit, order = 2)

## --- White's general test: replicate open_econs auxiliary exactly ---
u2 <- res^2
x1 <- df$x1
x2 <- df$x2
Z <- cbind(x1, x2, x1^2, x2^2, x1 * x2)
Z <- scale(Z, center = TRUE, scale = FALSE)  # mean-center each term
aux <- lm(u2 ~ Z)                             # auxiliary intercept included
white_r2 <- summary(aux)$r.squared
white_stat <- n * white_r2
white_df <- 5
white_pvalue <- pchisq(white_stat, df = white_df, lower.tail = FALSE)

## --- Ljung-Box (lag 1) ---
lb <- Box.test(res, lag = 1, type = "Ljung-Box")

## --- Influence measures (full vectors) ---
cooks <- as.numeric(cooks.distance(fit)); names(cooks) <- NULL
hat   <- as.numeric(hatvalues(fit));      names(hat)   <- NULL

dfb <- dfbetas(fit)                       # n x 3 matrix (Intercept, x1, x2)
dfb_cols <- colnames(dfb)
# serialize row-major as a list of rows so JSON round-trips to an (n, 3) array
dfb_rows <- lapply(seq_len(nrow(dfb)), function(i) as.numeric(dfb[i, ]))

out <- list(
  n = n,
  bg = list(
    order1 = list(lm_stat = unname(bg1$statistic), lm_pvalue = unname(bg1$p.value)),
    order2 = list(lm_stat = unname(bg2$statistic), lm_pvalue = unname(bg2$p.value))
  ),
  white = list(
    white_stat = white_stat,
    white_pvalue = white_pvalue,
    df = white_df
  ),
  ljung_box = list(
    lb_stat = unname(lb$statistic),
    lb_pvalue = unname(lb$p.value)
  ),
  cooks_distance = cooks,
  leverage = hat,
  dfbetas_cols = dfb_cols,
  dfbetas = dfb_rows
)

write_json(out, out_json, auto_unbox = TRUE, digits = 15, pretty = TRUE)
cat("diag_estat.R: wrote", out_json, "\n")
