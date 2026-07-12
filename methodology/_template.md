---
# Internal metadata — do not edit by hand.
# This template documents the canonical sections for every methodology page.
# Copy this file to create a new methodology page, then fill each section.
method: estimator_name
aliases:
  - alternative names
category: linear | causal_inference | decomposition | discrete | nonlinear | panel | diagnostics
api:
  - oe.estimator()
context_api:
  - ctx.estimator()
panel_api: []
problem: concise problem statement
estimator: estimator description
stata_equivalent:
  - equivalent Stata commands
r_equivalent:
  - equivalent R functions
status: mature | experimental
tier: 1 | 2 | 3
references:
  - BibTeXKey1
  - BibTeXKey2
---

# Estimator Title — SEO-Friendly Human Readable Name

> **Estimator summary**: one sentence explaining what this estimator does.

## Overview

What problem does this estimator solve? What does open-econs implement?

Keep this concise — 2–3 paragraphs maximum.

## Mathematical Formulation

### Population Model

\[
Y_i = X_i \beta + \varepsilon_i
\]

Define all terms. State the identification condition (e.g. `E[ε|X] = 0`).

### Estimator

\[
\hat{\beta} = (X'X)^{-1} X' Y
\]

Describe how the estimator is computed. Reference any closed-form or numerical solver used.

### Key Quantities of Interest

- Coefficients β
- Predicted values Ŷ
- Residuals e = Y − Ŷ
- Model fit statistics (R², adjusted R², F-statistic, log-likelihood, AIC, BIC)

## Assumptions

List all assumptions required for the estimator, using precise mathematical notation where applicable.

1. **Linearity**: The conditional mean is linear in parameters.
2. **Exogeneity**: E[ε | X] = 0 (for consistency).
3. **No perfect collinearity**: The design matrix X has full column rank.
4. **Conditional homoskedasticity** (for classical SEs): Var[ε | X] = σ² I.

Specify which assumptions are required for consistency, which are required for the default inference, and which can be relaxed.

## Estimator Derivation

Show the derivation path from the population model to the sample estimator:

\[
\hat{\beta} = \arg\min_\beta \sum_i (Y_i - X_i\beta)^2
\]

The first-order condition:

\[
-2 X'(Y - X\hat{\beta}) = 0 \quad \Rightarrow \quad X'X\hat{\beta} = X'Y
\]

The solution:

\[
\hat{\beta} = (X'X)^{-1} X' Y
\]

## Inference

### Covariance Estimators

Document every covariance estimator the implementation provides, with the exact formula used.

| Estimator | Formula | Use Case | Reference |
|-----------|---------|----------|-----------|
| Classical | `σ² (X'X)⁻¹` with `σ² = SSR / (n−k)` | Homoskedastic only | — |
| HC0 | `(X'X)⁻¹ X' diag(eᵢ²) X (X'X)⁻¹` | Heteroskedastic-robust | White (1980) |
| HC1 | `(n/(n−k)) · HC0` | Finite-sample correction | MacKinnon & White (1985) |
| HC2 | `(X'X)⁻¹ X' diag(eᵢ²/(1−hᵢᵢ)) X (X'X)⁻¹` | Leverage-adjusted | MacKinnon & White (1985) |
| HC3 | `(X'X)⁻¹ X' diag(eᵢ²/(1−hᵢᵢ)²) X (X'X)⁻¹` | Conservative | MacKinnon & White (1985) |
| Cluster | Sandwich clustered by group | Correlated errors within clusters | Liang & Zeger (1986); Cameron, Gelbach & Miller (2011) |
| Multi-way cluster | Minik estimator combining cluster dimensions | Multiple clustering dimensions | Cameron, Gelbach & Miller (2011) |
| HAC (Newey-West) | Bartlett kernel long-run variance | Autocorrelated time-series | Newey & West (1987) |

For each estimator:

- Describe the formula intuition.
- State the reference.
- Describe the use case.
- Note any differences from textbook formulations.

### Default Behavior

State the default `cov_type` and explain the choice.

### Technical Deviations from External Software

Document any parameterization or default differences from Stata, R, or statsmodels. For example:

- "open-econs defaults to HC2; Stata `reg, robust` uses HC1."
- "open-econs uses z-based (normal) inference for multi-way cluster and HAC; Stata uses t-based with various df approximations."
- "Newey-West HAC in open-econs defaults to the original NW1987 formula (no N/(N−K) adjustment); Stata's `newey` applies the adjustment unconditionally."

## Implementation Details

### Formula Interface

The formula uses the [formulaic](https://github.com/matthewwardrop/formulaic) library, supporting standard R-style formulas:

```
y ~ x1 + x2
y ~ x1 + x2 + x3 - 1        # no intercept
y ~ x1 + C(factor_var)      # categorical encoding
y ~ x1 | x2 | endog ~ instruments  # IV syntax (shared with iv())
```

### Result Object

The result object is **immutable** after construction.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.coefficients` | `pd.Series` | Coefficient estimates, named by variable |
| `.std_errors` | `pd.Series` | Standard errors |
| `.t_stats` | `pd.Series` | t-statistics |
| `.p_values` | `pd.Series` | p-values (normal-based for non-standard cov) |
| `.conf_int` | `pd.DataFrame` | 95% confidence intervals |
| `.r_squared` | `float` | R-squared |
| `.adj_r_squared` | `float` | Adjusted R-squared |
| `.f_statistic` | `float` | F-statistic |
| `.f_p_value` | `float` | F p-value |
| `.condition_number` | `float` | Scaled condition number (Belsley, Kuh & Welsch, 1980) |
| `.nobs` | `int` | Number of observations |
| `.df_resid` | `int` | Residual degrees of freedom |
| `.df_model` | `int` | Model degrees of freedom |
| `.llf` | `float` | Log-likelihood |
| `.aic` | `float` | Akaike Information Criterion |
| `.bic` | `float` | Bayesian Information Criterion |
| `.rsd` | `float` | Residual standard deviation |
| `.fitted_values` | `pd.Series` | Fitted values |
| `.residuals` | `pd.Series` | Residuals |
| `.cov_type` | `str` | Label for the covariance estimator used |
| `.call` | `dict` | Captured call arguments (for reproducibility) |

| Method | Description |
|--------|-------------|
| `.tidy()` | Coefficient table (DataFrame) |
| `.summary()` | Pretty-printed results with diagnostics |
| `.vcov()` | Variance-covariance matrix |
| `.predict(newdata)` | Predict (in-sample or on new data) |
| `.export(path)` | Export to JSON or CSV |
| `.to_latex()` / `.to_html()` | Export to LaTeX / HTML |
| `.wald_test(r_matrix)` | Wald test (via statsmodels) |
| `.f_test(r_matrix)` | F-test (via statsmodels) |
| `.diagnostics()` | Dictionary of Jarque-Bera, Breusch-Pagan, Durbin-Watson, Ramsey RESET |
| `.plot()` | Deprecated diagnostic plots |

### Diagnostics

The `.diagnostics()` method returns:

- **Jarque-Bera** test for normality of residuals
- **Breusch-Pagan** test for heteroskedasticity
- **Durbin-Watson** statistic for autocorrelation
- **Ramsey RESET** test for functional form misspecification (powers 2–3)

### Numerical Checks

- **Collinearity detection**: Condition number computed on the column-scaled (non-intercept) design matrix. A warning is issued when the condition number exceeds 30 (Belsley, Kuh & Welsch, 1980).
- **Singular matrix**: `np.linalg.matrix_rank` check raises an error if `rank(X) < k`.
- **Missing data**: Rows with any NaN in the formula variables are dropped with a warning.

## Stata / R Equivalents

### Stata

| open-econs | Stata | Notes |
|------------|-------|-------|
| `oe.ols("y ~ x1 + x2", data=df)` | `reg y x1 x2` | HC2 default vs Stata classical default |
| `oe.ols("y ~ x1 + x2", data=df, cov_type="HC1")` | `reg y x1 x2, robust` | Matches Stata's `vce(robust)` |
| `oe.ols("y ~ x1 + x2", data=df, cluster="cl")` | `reg y x1 x2, vce(cluster cl)` | Cluster SE (one-way) |
| `oe.ols("y ~ x1 + x2", data=df, cluster=["a", "b"])` | `reghdfe y x1 x2, cluster(a b)` | Multi-way cluster (CGM 2011) |
| `oe.ols("y ~ x1 + x2", data=df, cov_type="HAC", lags=2, time="t", hac_adjust=True)` | `newey y x1 x2, lag(2)` | HAC with Stata's N/(N−K) adjustment |

### R

| open-econs | R | Notes |
|------------|---|-------|
| `oe.ols("y ~ x1 + x2", data=df)` | `lm(y ~ x1 + x2, data=df)` | — |
| `oe.ols("y ~ x1 + x2", data=df, cov_type="HC2")` | `lmtest::coeftest(lm(y~x1+x2), vcov=vcovHC(type="HC2"))` | Both default to HC2 |
| `oe.ols("y ~ x1 + x2", data=df, cluster="cl")` | `lmtest::coeftest(lm(y~x1+x2), vcov=vcovCL(cluster=~cl))` | — |
| `oe.ols("y ~ x1 + x2", data=df, cov_type="HAC", lags=2)` | `lmtest::coeftest(lm(y~x1+x2), vcov=NeweyWest(lag=2, adjust=FALSE))` | Both default to no adjustment |

## API Examples

### Basic OLS

```python
import open_econs as oe

result = oe.ols("income ~ education + age", data=df)
print(result.tidy())
print(result.summary())
```

### Robust Standard Errors (HC1, matching Stata `reg, robust`)

```python
result = oe.ols("income ~ education + age", data=df, cov_type="HC1")
```

### Clustered Standard Errors

```python
# One-way cluster
result = oe.ols("income ~ education + age", data=df, cluster="province")

# Two-way cluster
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
    hac_adjust=True,  # optional N/(N-K) correction
)
```

### Weighted Least Squares

```python
result = oe.ols(
    "income ~ education + age",
    data=df,
    weights="population",
    cov_type="nonrobust",
)
```

### Context API

```python
ctx = oe.Context(df)
r1 = ctx.ols("income ~ education + age")
r2 = ctx.ols("income ~ education + age", cluster="province")
```

## Limitations

1. **No LASSO / Ridge / ElasticNet**: Regularized regression is not implemented.
2. **No Quantile Regression**: Only conditional mean estimation.
3. **No M-estimators**: Robust regression via Huber/M-estimation is not supported.
4. **No FWL theorem for high-dimensional FE**: For multi-way fixed effects beyond entity+time, use `oe.fe()`.
5. **No bootstrap SEs**: Standard errors are analytic; bootstrap is not available for OLS.
6. **Weights + multi-way cluster / HAC**: Weights are not supported together with multi-way clustering or Newey-West HAC.

## References
