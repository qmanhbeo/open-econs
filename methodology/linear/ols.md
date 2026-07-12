---
method: ols
aliases:
  - linear regression
  - ordinary least squares
  - WLS
category: linear
api:
  - oe.ols()
  - oe.reg()
  - ctx.ols()
panel_api: []
panel_context_api: []
problem: conditional mean estimation
estimator: ordinary least squares
stata_equivalent:
  - regress
  - reg
  - newey
r_equivalent:
  - lm
status: mature
tier: 1
references:
  - white1980
  - mackinnonwhite1985
  - neweywest1987
  - camerongelbachmiller2011
  - belsleykuhwelsch1980
  - liangzeger1986
---

# OLS Regression with Robust and Clustered Standard Errors in Python

> **Estimator summary**: open-econs implements Ordinary Least Squares via statsmodels with custom support for multi-way clustered errors (Cameron, Gelbach & Miller 2011) and Newey-West HAC standard errors, defaulting to HC2 robust covariance.

## Overview

OLS estimates the conditional mean `E[Y | X]` under a linear specification. The open-econs `ols()` function wraps statsmodels for the core OLS fit, then replaces the variance-covariance computation for multi-way clustering and HAC with custom implementations.

Three distinct code paths exist inside `ols()`:

- **Standard HC0–HC3 / single cluster**: delegated to statsmodels `OLS.fit(cov_type=...)`
- **Multi-way cluster** (`cluster=["a", "b", ...]`): computed via the CGM (2011) minik estimator with manual inclusion-exclusion over cluster intersections
- **Newey-West HAC**: computed via `newey_west_cov()` using a Bartlett kernel with optional panel-cluster aggregation

## Mathematical Formulation

### Population Model

\[
Y_i = X_i \beta + \varepsilon_i, \quad i = 1, \dots, n
\]

where `Y_i` is a scalar outcome, `X_i` is a `1 × k` row vector of regressors (including a constant), and `ε_i` is an unobserved error term.

### OLS Estimator

\[
\hat{\beta} = (X'X)^{-1} X' Y = \arg\min_\beta \sum_{i=1}^n (Y_i - X_i\beta)^2
\]

### Weighted Least Squares

When `weights` are provided, the estimator minimises:

\[
\hat{\beta}_{\text{WLS}} = (X'W X)^{-1} X' W Y, \quad W = \text{diag}(w_i)
\]

where the WLS fit is delegated entirely to statsmodels `WLS.fit()` (only single-cluster or HC0–HC3 inference with weights).

### Key Quantities of Interest

- Coefficients `β̂`
- Standard errors `SE(β̂_j)` — type depends on `cov_type`
- t-statistics `t_j = β̂_j / SE(β̂_j)`
- p-values (normal-based for multi-way cluster and HAC; t-based for statsmodels-backed paths)
- 95% confidence intervals `β̂_j ± 1.96 × SE(β̂_j)` (normal-based for multi-way cluster and HAC; t-based otherwise)
- R², adjusted R², F-statistic (robust Wald F for non-classical cov types), log-likelihood, AIC, BIC
- Condition number (scaled, no-intercept)

## Assumptions

1. **Linearity**: `E[Y | X] = Xβ` (linear in parameters).
2. **Exogeneity**: `E[ε_i | X] = 0` for all `i` (required for consistency of β̂).
3. **No perfect collinearity**: `rank(X) = k` (the design matrix is full column rank).
4. **Conditional homoskedasticity + no autocorrelation**: `Var[ε | X] = σ² I` (required only for classical `cov_type="nonrobust"` inference; relaxed by all robust/cluster/HAC estimators).
5. **Independent errors across clusters** (for cluster-robust SEs).
6. **Stationarity + weak dependence** (for Newey-West HAC consistency).

## Estimator Derivation

The OLS estimator is obtained by solving the normal equations:

\[
\frac{\partial}{\partial \beta} \sum_i (Y_i - X_i\beta)^2 = -2 X'(Y - X\beta) = 0
\]

\[
X'X\hat{\beta} = X'Y
\]

\[
\hat{\beta} = (X'X)^{-1} X'Y
\]

## Inference

### Covariance Estimators

| Estimator | Formula | Use Case | Reference |
|-----------|---------|----------|-----------|
| Classical (`nonrobust`) | `σ̂² (X'X)⁻¹`, `σ̂² = e'e / (n−k)` | Homoskedastic iid errors | — |
| HC0 | `(X'X)⁻¹ X' diag(eᵢ²) X (X'X)⁻¹` | Heteroskedastic-robust | @white1980 |
| HC1 | `(n/(n−k)) · HC0` | Stata `reg, robust` default | @mackinnonwhite1985 |
| HC2 | `(X'X)⁻¹ X' diag(eᵢ²/(1−hᵢᵢ)) X (X'X)⁻¹` | Leverage-adjusted, **default** | @mackinnonwhite1985 |
| HC3 | `(X'X)⁻¹ X' diag(eᵢ²/(1−hᵢᵢ)²) X (X'X)⁻¹` | Conservative (drop-1 jackknife approx) | @mackinnonwhite1985 |
| Cluster (single) | Sandwich clustered by `g∈{1,…,G}` | Correlated errors within groups | @liangzeger1986 |
| Multi-way cluster | `V_{g1}+V_{g2}−V_{g1∩g2}` (inclusion-exclusion minik) | Multiple clustering dimensions | @camerongelbachmiller2011 |
| HAC (Newey-West) | `(X'X)⁻¹ Ŝ (X'X)⁻¹` with Bartlett kernel | Autocorrelated time-series errors | @neweywest1987 |

**Notes on the default (`HC2`)**: open-econs defaults to `cov_type="HC2"`, matching R's `sandwich::vcovHC(type="HC2")` and modern Stata (which adopted HC2 in Stata 14+ for `reg, vce(robust)`). Original Stata `reg, robust` (pre-Stata 14) used HC1.

### Multi-way Cluster Implementation

The multi-way cluster variance follows Cameron, Gelbach & Miller (2011), the "minik" estimator. For two cluster dimensions `g1` and `g2`:

\[
V_{\text{multi}} = V_{g1} + V_{g2} - V_{g1∩g2}
\]

where each `V_g` is the standard Liang-Zeger cluster variance using group `g`. For three or more dimensions, the formula generalises to inclusion-exclusion over all non-empty subsets:

\[
V = \sum_{s=1}^{|S|} (-1)^{s+1} \sum_{C \subseteq S, |C|=s} V_{\cap_{j\in C} g_j}
\]

The `_minik_contribution` function sum-of-outer-products within each cluster level, combined over intersections using composite integer labels (`label1 × (max(label2)+2) + label2`).

### Newey-West HAC Implementation

The long-run covariance is computed with a Bartlett kernel:

\[
\hat{S} = \hat{\Gamma}_0 + \sum_{j=1}^{L} w(j, L) (\hat{\Gamma}_j + \hat{\Gamma}_j')
\]

\[
w(j, L) = 1 - \frac{j}{L+1}, \quad \hat{\Gamma}_j = \frac{1}{n} \sum_{t=j+1}^{n} e_t e_{t-j} X_t' X_{t-j}
\]

where `e_t` are OLS residuals and `L` is the `lags` parameter. When `hac_adjust=True`, the variance is multiplied by `n / (n − k)` (borrowed from HC1; not part of the original @neweywest1987 paper).

**Panel HAC**: When `cluster` is provided together with `cov_type="HAC"`, score contributions are first aggregated within clusters (summing per-cluster), and the HAC is computed on the cluster-level score vectors. This matches Stata's panel HAC (`newey Y X, force`).

### Default Behavior

- **Default `cov_type`**: `"HC2"`
- **Inference for multi-way cluster and HAC**: z-based (normal approximation), not t-based. Standard errors use 1.96 for 95% confidence intervals. This differs from Stata and statsmodels, which use t-distributions with various degrees-of-freedom approximations.
- **F-statistic**: For robust cov types, statsmodels computes a robust Wald F-statistic (heteroskedasticity-consistent). The F-distribution approximation may differ from Stata's small-sample corrections.

### Technical Deviations from External Software

| Feature | open-econs | Stata | R |
|---------|------------|-------|---|
| Default robust SE | HC2 | HC1 (pre-14); HC2 (14+) | HC2 (`vcovHC`) |
| Multi-way cluster t-dist | z-based (normal) | `reghdfe` uses t with `min(G, N−k)` df | `vcovCL` uses z-based |
| HAC N/(N−K) adjustment | Off by default (`hac_adjust=False`) | Always on (`newey`) | Off by default (`adjust=FALSE`) |
| HAC lags | User-specified (required) | Default `floor(N^{1/4})` or `floor(T^{1/4})` | User-specified |
| Conf int for cluster SE | z-based (1.96) | t-based | t-based (various df) |
| Weights + cluster | Only single-cluster | Supports all combinations | Supports via `svyglm` |

## Implementation Details

### Formula Interface

Uses the [formulaic](https://github.com/matthewwardrop/formulaic) library, supporting standard R-style formulas. The formula is split at `~` into left-hand side (dependent variable) and right-hand side (regressors).

```python
"y ~ x1 + x2"              # basic
"y ~ x1 + x2 - 1"          # no intercept
"y ~ x1 + C(factor_var)"   # categorical encoding
```

The model matrix is built from the full formula object; if formulaic raises a missing-column error, a user-friendly `ValueError` with the available column names is raised instead.

### Result Object

Returns an `OLSResult` (immutable via `BaseModel._freeze()`). Key attributes and methods:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.coefficients` | `pd.Series` | Named coefficient array |
| `.std_errors` | `pd.Series` | Standard errors |
| `.t_stats` | `pd.Series` | t-statistics |
| `.p_values` | `pd.Series` | p-values |
| `.conf_int` | `pd.DataFrame` | Columns `lower`, `upper` |
| `.r_squared` | `float` | R-squared |
| `.adj_r_squared` | `float` | Adjusted R-squared |
| `.f_statistic` | `float` | Wald F-statistic (robust) |
| `.f_p_value` | `float` | F p-value |
| `.condition_number` | `float` | Scaled condition number |
| `.nobs` | `int` | N used after dropping NaN |
| `.df_resid` | `int` | Residual degrees of freedom |
| `.cov_type` | `str` | Label string for display |

| Method | Description |
|--------|-------------|
| `.tidy()` | Coefficient table as DataFrame |
| `.summary()` | Printed results string with diagnostics |
| `.vcov()` | Variance-covariance matrix |
| `.predict(newdata)` | Predict values |
| `.export(path)` | Save as `.json` or `.csv` |
| `.wald_test(r_matrix)` | Wald test |
| `.f_test(r_matrix)` | F-test |
| `.diagnostics()` | Dict: Jarque-Bera, Breusch-Pagan, Durbin-Watson, Ramsey RESET |

### Covariance Label Convention

The `.cov_type` attribute is set to a descriptive string:

| Code path | `cov_type` value |
|-----------|-----------------|
| `cov_type="nonrobust"` | `"nonrobust"` |
| `cov_type="HC0"`–`"HC3"` | `"HC0"`, `"HC1"`, etc. |
| `cluster="province"` | `"cluster(province)"` |
| `cluster=["firm", "year"]` | `"cluster(firm, year)"` |
| `cov_type="HAC", lags=2` | `"HAC(2)"` |
| `cov_type="HAC", lags=2, cluster="firm"` | `"HAC(2) cluster(firm)"` |

### Collinearity Check

The `_check_collinearity` function in `ols.py`:

1. Computes `matrix_rank(X)`. If `rank < k`, raises `RuntimeError("Singular design matrix")`.
2. Drops the Intercept column (if present), scales remaining columns by their standard deviations, and computes the condition number.
3. If the scaled condition number exceeds 30, a `RuntimeWarning` is issued citing Belsley, Kuh & Welsch (1980).

### Missing Data

Rows with any NaN in the formula variables are dropped with a `RuntimeWarning` listing which columns had NaN values. If zero rows remain, a descriptive error is raised.

### Diagnostics

Available via `result.diagnostics()`:

- **Jarque-Bera**: Test for normality of residuals (from `scipy.stats`).
- **Breusch-Pagan**: Test for heteroskedasticity (from `statsmodels.stats.diagnostic`), using the full design matrix.
- **Durbin-Watson**: Test for first-order autocorrelation (from `statsmodels.stats.stattools`).
- **Ramsey RESET**: Functional form test using powers 2 and 3 of fitted values (implemented in `results.py:_ramsey_reset`).

### Plot

`result.plot()` is **deprecated in v0.8** and scheduled for removal in v0.9.

## WLS (Weighted Least Squares)

When `weights` is specified:

- For `weights` as a string: interpreted as a column name in `data`.
- For `weights` as an array: must match the number of data rows.
- The fit delegates to statsmodels `WLS.fit()`.
- **Limitation**: Weights are not supported together with multi-way clustering or Newey-West HAC (raises `ValueError`).
- Negative weights raise `ValueError`.

## Stata / R Equivalents

### Stata

| open-econs | Stata | Notes |
|------------|-------|-------|
| `oe.ols("y ~ x1 + x2", data=df)` | `reg y x1 x2` | HC2 default vs Stata classical default |
| `oe.ols("y ~ x1 + x2", data=df, cov_type="HC1")` | `reg y x1 x2, robust` | Matches Stata's `vce(robust)` |
| `oe.ols("y ~ x1 + x2", data=df, cluster="cl")` | `reg y x1 x2, vce(cluster cl)` | Cluster SE |
| `oe.ols("y ~ x1 + x2", data=df, cluster=["a","b"])` | `reghdfe y x1 x2, cluster(a b)` | Multi-way cluster |
| `oe.ols("y ~ x1 + x2", data=df, cov_type="HAC", lags=2, time="t", hac_adjust=True)` | `newey y x1 x2, lag(2)` | HAC matching Stata |

### R

| open-econs | R | Notes |
|------------|---|-------|
| `oe.ols("y ~ x1 + x2")` | `lm(y ~ x1 + x2, data=df)` | — |
| `oe.ols("y ~ x1 + x2", cov_type="HC2")` | `lmtest::coeftest(lm(...), vcov=vcovHC(type="HC2"))` | Both default to HC2 |
| `oe.ols("y ~ x1 + x2", cluster="cl")` | `lmtest::coeftest(lm(...), vcov=vcovCL(cluster=~cl))` | — |
| `oe.ols("y ~ x1 + x2", cov_type="HAC", lags=2)` | `lmtest::coeftest(lm(...), vcov=NeweyWest(lag=2, adjust=F))` | Both default to no adjustment |

## API Examples

### Basic OLS

```python
import open_econs as oe

result = oe.ols("income ~ education + age", data=df)
print(result.tidy())
#       Variable      Coef    Std Err        t    P>|t|      0.025      0.975
# 0   Intercept  4.526102  0.050123  90.3014  0.00000  4.427861  4.624343
# 1   education  0.083172  0.004521  18.3936  0.00000  0.074311  0.092033
# 2         age  0.020013  0.001025  19.5275  0.00000  0.018004  0.022022
```

### Robust Standard Errors (HC1, matching Stata `reg, robust`)

```python
result = oe.ols("income ~ education + age", data=df, cov_type="HC1")
```

### Clustered Standard Errors

```python
# One-way
result = oe.ols("income ~ education + age", data=df, cluster="province")

# Two-way (CGM 2011)
result = oe.ols("income ~ education + age", data=df, cluster=["firm", "year"])
```

### Newey-West HAC

```python
result = oe.ols(
    "income ~ education + age",
    data=df,
    cov_type="HAC",
    lags=2,
    time="time",
)
```

### Newey-West HAC Matching Stata

```python
result = oe.ols(
    "income ~ education + age",
    data=df,
    cov_type="HAC",
    lags=2,
    time="time",
    hac_adjust=True,
)
```

### Weighted Least Squares

```python
result = oe.ols("income ~ education + age", data=df, weights="population")
```

### Context API

```python
ctx = oe.Context(df)
r1 = ctx.ols("income ~ education + age")
r2 = ctx.ols("income ~ education + age", cluster="province")
```

## Limitations

1. **No LASSO / Ridge / ElasticNet**: Only unregularised OLS.
2. **No Quantile Regression**: Only conditional mean.
3. **No M-estimators**: No Huber-White robust regression.
4. **Weights + multi-way cluster / HAC**: Not currently supported (raises `ValueError`).
5. **Weights + single cluster**: Delegated to statsmodels WLS (single-cluster only; multi-way not supported).
6. **No bootstrap inference**: All standard errors are analytic.
7. **No FWL residualization**: For absorbing high-dimensional fixed effects, use `fe()` instead.
8. **Normal-based CIs for multi-way cluster and HAC**: Uses 1.96 instead of t-based critical values with cluster-adjusted df.
9. **No Stata `newey` auto-lag selection**: The `lags` parameter is required (no `floor(N^{1/4})` default).
10. **`plot()` deprecated in v0.8**: Scheduled for removal in v0.9.

## References

- @white1980
- @mackinnonwhite1985
- @neweywest1987
- @camerongelbachmiller2011
- @belsleykuhwelsch1980
- @liangzeger1986
