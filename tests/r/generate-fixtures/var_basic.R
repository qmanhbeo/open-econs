# VAR/VECM fixture generation for open-econs parity tests
# Uses var_input.csv (200 obs, 2 variables: y1, y2)
#
# Johansen case correspondence (source-confirmed):
#   R ecdet="none"  -> Stata Case 3: unrestricted constant (trend(constant))
#   R ecdet="const" -> Stata Case 2: restricted constant   (trend(rconstant))
#   R ecdet="trend" -> Stata Case 4: restricted trend      (trend(rtrend))
#
# R cannot compute Stata Cases 1 (no deterministic) or 5 (unrestricted trend).
# These are marked as NA in the output.
#
# ca.jo @teststat is stored in DESCENDING rank order [r<=K-1,...,r=0].
# Stata e(trace)/e(max) is [r<=0,...,r<=K-1] (ascending).
# We reverse R vectors to match Stata convention before storing.

library(vars)
library(urca)
library(jsonlite)

# Import data
data <- read.csv("tests/r/fixtures/inputs/var_input.csv")
y <- as.matrix(data)

# ── 1. Lag order selection (VARselect) ──────────────────────────
sel <- VARselect(y, lag.max = 5, type = "const")

# ── 2. VAR estimation at lag 2 ──────────────────────────────────
var_fit <- VAR(y, p = 2, type = "const")

# ── 3. Johansen cointegration — R can compute 3 of the 5 Stata cases ──
# ecdet="none"  computes Stata Case 3 (unrestricted constant)
jo_trace_none  <- ca.jo(y, type = "trace", ecdet = "none",  K = 2)
jo_eigen_none  <- ca.jo(y, type = "eigen", ecdet = "none",  K = 2)

# ecdet="const" computes Stata Case 2 (restricted constant)
jo_trace_const <- ca.jo(y, type = "trace", ecdet = "const", K = 2)
jo_eigen_const <- ca.jo(y, type = "eigen", ecdet = "const", K = 2)

# ecdet="trend" computes Stata Case 4 (restricted trend)
jo_trace_trend <- ca.jo(y, type = "trace", ecdet = "trend", K = 2)
jo_eigen_trend <- ca.jo(y, type = "eigen", ecdet = "trend", K = 2)

# ── 4. Granger causality ───────────────────────────────────────
gc_test <- causality(var_fit, cause = "y1")

# ── 5. Save results as JSON ────────────────────────────────────

# Helper: reverse ca.jo @teststat to Stata order [r<=0, r<=1]
to_stata_order <- function(jo_obj) rev(as.numeric(jo_obj@teststat))

# Helper: extract 5% CV column from ca.jo cval matrix (col 2 = 5%)
# ca.jo @cval rows are in DESCENDING rank order; reverse to match Stata.
cv5_trace <- function(jo_obj) rev(as.numeric(jo_obj@cval[, 2]))
cv5_eigen <- function(jo_obj) rev(as.numeric(jo_obj@cval[, 2]))

results <- list(
  # Lag selection
  selected_lag = list(
    aic = sel$selection["AIC(n)"],
    hqic = sel$selection["HQ(n)"],
    bic = sel$selection["SC(n)"],
    fpe = sel$selection["FPE(n)"]
  ),

  # Case 1 (Stata: trend(none)) — NOT computable in R
  trace_case1 = NA,
  cv_trace5_case1 = NA,
  maxeig_case1 = NA,
  cv_maxeig5_case1 = NA,

  # Case 2 (Stata: trend(rconstant)) — R ecdet="const"
  trace_case2 = to_stata_order(jo_trace_const),
  cv_trace5_case2 = cv5_trace(jo_trace_const),
  maxeig_case2 = to_stata_order(jo_eigen_const),
  cv_maxeig5_case2 = cv5_eigen(jo_eigen_const),

  # Case 3 (Stata: trend(constant)) — R ecdet="none"
  trace_case3 = to_stata_order(jo_trace_none),
  cv_trace5_case3 = cv5_trace(jo_trace_none),
  maxeig_case3 = to_stata_order(jo_eigen_none),
  cv_maxeig5_case3 = cv5_eigen(jo_eigen_none),

  # Case 4 (Stata: trend(rtrend)) — R ecdet="trend"
  trace_case4 = to_stata_order(jo_trace_trend),
  cv_trace5_case4 = cv5_trace(jo_trace_trend),
  maxeig_case4 = to_stata_order(jo_eigen_trend),
  cv_maxeig5_case4 = cv5_eigen(jo_eigen_trend),

  # Case 5 (Stata: trend(trend)) — NOT computable in R
  trace_case5 = NA,
  cv_trace5_case5 = NA,
  maxeig_case5 = NA,
  cv_maxeig5_case5 = NA,

  # Granger causality (y1 -> y2)
  granger_f_stat = gc_test$Granger$statistic,
  granger_f_pvalue = gc_test$Granger$p.value,
  granger_f_df1 = gc_test$Granger$parameter[1],
  granger_f_df2 = gc_test$Granger$parameter[2],

  # Instantaneous causality
  instant_chi2 = gc_test$Instant$statistic,
  instant_pvalue = gc_test$Instant$p.value,
  instant_df = gc_test$Instant$parameter
)

# Convert matrix columns to plain vectors for JSON serialization
for (nm in names(results)) {
  if (is.matrix(results[[nm]])) {
    results[[nm]] <- as.vector(results[[nm]])
  }
}

write_json(results, "tests/r/fixtures/expected/var_basic.json", pretty = TRUE, auto_unbox = TRUE)

cat("R fixtures written to tests/r/fixtures/expected/var_basic.json\n")
cat("Selected lags: AIC=", results$selected_lag$aic,
    " BIC=", results$selected_lag$bic, "\n")
cat("Case 2 trace (Stata rconstant):", results$trace_case2, "\n")
cat("Case 3 trace (Stata constant):", results$trace_case3, "\n")
cat("Case 4 trace (Stata rtrend):", results$trace_case4, "\n")
cat("Granger F=", results$granger_f_stat, " p=", results$granger_f_pvalue, "\n")
