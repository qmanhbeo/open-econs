# ARDL/UECM + PSS bounds-test fixture generation for open-econs parity tests
# Uses ardl_input.csv (55 rows, 5 cols: LRM, LRY, LPY, IBO, IDE).
# Canonical example: R ARDL denmark data, LRM ~ LRY + IBO + IDE,
# order=c(3,1,3,2), PSS case 3.
#
# Source-verified conventions:
#   - uecm()/ardl() accept a ts object built from the denmark-equivalent CSV.
#   - bounds_f_test(uecm_model, case=3)$statistic  -> F (Wald) statistic
#   - bounds_t_test(uecm_model, case=3)$statistic  -> t statistic
#   - EC term = coef on L(LRM, 1) in the uecm object
#   - multipliers(uecm_model)$Estimate indexed by Term (LRY/IBO/IDE)
#   - critical values from ARDL:::crit_val_bounds_pss2001$f$iii and $t$iii
#     (case 3 = "iii"), k = number of regressors = 3.

library(ARDL)
library(jsonlite)

# Import data (byte-identical to R data(denmark); LPY unused)
data0 <- read.csv("tests/r/fixtures/inputs/ardl_input.csv")
data <- ts(data0, start = 1974, frequency = 4)

# ── 1. UECM and ARDL estimation ───────────────────────────────
uecm_model <- uecm(LRM ~ LRY + IBO + IDE, data = data, order = c(3, 1, 3, 2))
ardl_model <- ardl(LRM ~ LRY + IBO + IDE, data = data, order = c(3, 1, 3, 2))

# ── 2. Bounds F-test (case 3) ────────────────────────────────
bf <- bounds_f_test(uecm_model, case = 3)
f_stat <- as.numeric(bf$statistic)

# ── 3. Bounds t-test (case 3) ────────────────────────────────
bt <- bounds_t_test(uecm_model, case = 3)
t_stat <- as.numeric(bt$statistic)

# ── 4. Error-correction term = coef on L(LRM, 1) ─────────────
ec_term <- as.numeric(coef(uecm_model)[["L(LRM, 1)"]])

# ── 5. Long-run multipliers ──────────────────────────────────
ml <- multipliers(uecm_model)
ml_est <- setNames(ml$Estimate, ml$Term)
lr_LRY <- as.numeric(ml_est[["LRY"]])
lr_IBO <- as.numeric(ml_est[["IBO"]])
lr_IDE <- as.numeric(ml_est[["IDE"]])

# ── 6. PSS(2001) critical values, case 3 (iii), k = 3 ────────
cv <- ARDL:::crit_val_bounds_pss2001
kk <- 3
fiii <- cv$f$iii[cv$f$iii$k == kk, ]
tiii <- cv$t$iii[cv$t$iii$k == kk, ]

f_row <- function(a) {
  r <- fiii[fiii$alpha == a, ]
  c(lower = r$I0, upper = r$I1)
}
t_row <- function(a) {
  r <- tiii[tiii$alpha == a, ]
  c(lower = r$I0, upper = r$I1)
}

f10 <- f_row(0.10)
f05 <- f_row(0.05)
f01 <- f_row(0.01)
t10 <- t_row(0.10)
t05 <- t_row(0.05)
t01 <- t_row(0.01)

# ── 7. Save results as JSON ──────────────────────────────────
results <- list(
  f_stat = f_stat,
  t_stat = t_stat,
  ec_term = ec_term,
  lr_LRY = lr_LRY,
  lr_IBO = lr_IBO,
  lr_IDE = lr_IDE,
  f_cv_lower_10 = f10["lower"],
  f_cv_upper_10 = f10["upper"],
  f_cv_lower_5  = f05["lower"],
  f_cv_upper_5  = f05["upper"],
  f_cv_lower_1  = f01["lower"],
  f_cv_upper_1  = f01["upper"],
  t_cv_lower_10 = t10["lower"],
  t_cv_upper_10 = t10["upper"],
  t_cv_lower_5  = t05["lower"],
  t_cv_upper_5  = t05["upper"],
  t_cv_lower_1  = t01["lower"],
  t_cv_upper_1  = t01["upper"]
)

write_json(results, "tests/r/fixtures/expected/ardl.json",
           pretty = TRUE, auto_unbox = TRUE, digits = 15)

cat("R fixtures written to tests/r/fixtures/expected/ardl.json\n")
cat("F-stat=", results$f_stat, "\n")
cat("t-stat=", results$t_stat, "\n")
cat("EC term=", results$ec_term, "\n")
cat("LR LRY=", results$lr_LRY, " IBO=", results$lr_IBO, " IDE=", results$lr_IDE, "\n")
cat("F CV 10% (", results$f_cv_lower_10, ",", results$f_cv_upper_10, ")\n")
cat("F CV 5%  (", results$f_cv_lower_5, ",", results$f_cv_upper_5, ")\n")
cat("F CV 1%  (", results$f_cv_lower_1, ",", results$f_cv_upper_1, ")\n")
cat("t CV 10% (", results$t_cv_lower_10, ",", results$t_cv_upper_10, ")\n")
cat("t CV 5%  (", results$t_cv_lower_5, ",", results$t_cv_upper_5, ")\n")
cat("t CV 1%  (", results$t_cv_lower_1, ",", results$t_cv_upper_1, ")\n")
