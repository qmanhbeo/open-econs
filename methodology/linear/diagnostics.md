# OLS Regression Diagnostics in open-econs — Breusch-Godfrey, White, Ljung-Box, Cook's Distance, Leverage & DFBETAS (Stata/R Parity) {#diagnostics}

A practical, source-verified reference for the post-estimation diagnostics
shipped with `open_econs` v1.3.0. Every statistic is implemented from the
reference convention (Stata `estat` family / R `stats::`) or wrapped from a
working `statsmodels` backend, and cross-checked within the 1e-6 numeric
tolerance of the underlying math — never against output alone (AGENTS.md rule 1
& 2).

## Math & statistics

Let `X` be the full design matrix with an intercept column, `beta` the OLS
coefficients, `resid = y - X beta` the residuals, `n` the sample size, and
`k` the number of parameters (including the constant). The hat matrix is

```
H = X (X'X)^{-1} X' ,   h_ii = H_{ii}  (leverage of observation i)
```

and `s^2 = resid' resid / (n - k)`.

### Breusch-Godfrey (autocorrelation) — `bg_test`

Auxiliary regression of `resid` on the **full** design matrix (constant +
regressors) and `lags` lagged residuals:

```
resid_t = alpha' X_t + sum_{j=1}^{lags} gamma_j resid_{t-j} + error_t
```

The LM statistic is

```
LM = n * R^2  ~  chi2(lags)
```

and an F version (reported by Stata `estat bgodfrey`) is also returned. The
degrees of freedom are exactly `lags`.

### White (heteroskedasticity) — `white_test`

Auxiliary OLS of `resid^2` on the **regressors only** (the constant is
excluded), their squares, and (with `interaction=True`) all pairwise
cross-products, each term mean-centered (Stata `estat imtest, white`
convention):

```
resid^2 = a + sum_j b_j x_j + sum_j c_j x_j^2 + sum_{a<b} d_{ab} x_a x_b + error
```

with `p` non-constant regressors. The LM statistic is

```
LM = n * R^2  ~  chi2(df),   df = p + p(p+1)/2
```

For `p = 2` (two regressors) `df = 2 + 3 = 5`. **Note:** the constant is
excluded from the count, and Stata spells this as `estat imtest, white` — there
is **no** `estat hettest, white`.

### Ljung-Box — `ljung_box`

Wraps `statsmodels.stats.diagnostic.acorr_ljungbox` (this backend is intact):

```
Q = n(n+2) sum_{j=1}^{lags} (hat{rho}_j^2 / (n - j))  ~  chi2(lags)
```

`box_pierce=True` additionally returns the Box-Pierce statistic. Residuals are
mean-zero by construction.

### Cook's distance — `cooks_distance`

```
D_i = (t_i^2 / k) * (h_ii / (1 - h_ii))
```

where `t_i = resid_i / (sqrt(1 - h_ii) sqrt(s^2))` is the internally
studentized residual. Matches Stata `predict, cooksd` and R
`cooks.distance`.

### Leverage — `leverage`

`h_ii`, the diagonal of the hat matrix `H`. Matches Stata `predict, leverage`
and R `hatvalues`.

### DFBETAS / DFBETA — `dfbetas`, `dfbeta`

For parameter `j` and observation `i`:

```
DFBETA_{ij}  = b_j - b_{j(-i)}                       (raw, Stata `dfbeta`)
DFBETAS_{ij} = (b_j - b_{j(-i)}) / SE_j(-i)          (standardized)
```

where `b_{j(-i)}` is the leave-one-out coefficient and `SE_j(-i)` is the
leave-one-out standard error. `dfbetas()` returns the **standardized** form
(matching R `stats::dfbetas` and Stata's `predict, dfbeta` standardization);
`dfbeta()` returns the **raw** difference (matching Stata's `dfbeta` command,
which drops the constant by default — oe returns all parameters so callers can
slice). Internally, the leave-one-out coefficient uses the standard updating
formula `b(-i) = b - (X'X)^{-1} X_i' e_i / (1 - h_ii)`.

### Studentized residuals

- **Internally** studentized: `t_i = e_i / (sqrt(1 - h_ii) sqrt(s^2))`.
- **Externally** studentized (`influence()["resid_studentized"]`): uses the
  leave-one-out variance `s_{(-i)}^2`, matching Stata/R `rstudent`.
- **DFFITS** (`influence()["dffits"]`): `rstudent_i * sqrt(h_ii / (1 - h_ii))`.

## Usage

```python
import open_econs as oe

r = oe.ols("y ~ x1 + x2", data=df)   # cov_type defaults to "nonrobust"

# Autocorrelation
bg = r.bg_test(lags=1)               # {"lm_stat","lm_pvalue","f_stat","f_pvalue","df"}

# Heteroskedasticity (White)
white = r.white_test(interaction=True)   # {"white_stat","white_pvalue","df"}

# Portmanteau test on residuals
lb = r.ljung_box(lags=1, box_pierce=False)  # {"lb_stat","lb_pvalue"}

# Influence per observation
cooks = r.cooks_distance()           # pd.Series
lev   = r.leverage()                 # pd.Series
dbeta_s = r.dfbetas()                # pd.DataFrame (standardized)
dbeta_r = r.dfbeta()                 # pd.DataFrame (raw)

# One-shot bundle
inf = r.influence()                  # dict: cooks_distance, leverage,
                                     #       dfbetas, resid_studentized, dffits

# Summaries
table = r.diagnostics_table()        # pd.DataFrame (full battery)
diag  = r.diagnostics()              # dict (legacy JB/BP/DW/RESET form)
```

## Root-cause & footgun notes (AGENTS.md rule 16)

- **`statsmodels.acorr_breusch_godfrey` is broken in this environment**
  (statsmodels 0.14.6 dropped the design-matrix argument and runs the auxiliary
  regression on lagged residuals only, which does NOT match Stata). `bg_test`
  is therefore implemented **from scratch** — reconstruct the auxiliary
  regression in numpy and confirm `n * R^2` to validate. Do not reintroduce the
  statsmodels wrapper.
- **Stata has no `estat hettest, white`.** The White test is `estat imtest,
  white`. The White `df` counts only the non-constant regressors; with `p`
  regressors `df = p + p(p+1)/2`.
- **Stata `dfbeta` returns RAW deltas; `dfbetas` (R) is standardized.** oe
  exposes `dfbeta()` (raw) and `dfbetas()` (standardized) to cover both. Stata's
  `dfbeta` drops the constant by default; oe keeps all columns.
- **`diagnostics()` returns a dict while `diagnostics_table()` returns a
  DataFrame.** The dict form is retained for backward compatibility (an
  existing test asserts dict). See the roadmap note about flipping
  `diagnostics()` to a DataFrame in a future minor.
- **DFBETAS convention (RESOLVED):** oe standardizes by the leave-one-out
  variance `s_{(-i)}^2 = (RSS - e_i^2/(1-h_ii)) / (n-k-1)`, matching R
  `stats::dfbetas` and Stata `predict, dfbeta`. `statsmodels.OLSInfluence.dfbetas`
  uses the *same* leave-one-out variance (`sigma2_not_obsi`); the two now agree to
  machine precision (~9e-14), so the former ~8.6e-4 divergence recorded in
  FUTURE_WORK is gone (the LOO-variance factor `1/(1-h_i)` was fixed; see
  `tests/r/tests/test_r_diagnostics.py::test_dfbetas_gap_magnitude`). The
  `dfbetas(backend=...)` toggle (default `"stata_r"`, alternative
  `"statsmodels"`) exposes the convention choice per AGENTS.md rule 15 so the
  reference is explicit and auditable, not hidden. The default is validated
  against the authoritative Stata/R fixture (not statsmodels) in
  `tests/non_stata_nor_r/test_diagnostics.py`.
