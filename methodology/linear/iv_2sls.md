---
method: iv_2sls
aliases:
  - instrumental variables
  - two-stage least squares
  - 2SLS
  - IV
category: linear
api:
  - oe.iv()
context_api: []
panel_api: []
panel_context_api: []
problem:
  - endogenous regressors
  - causal identification
estimator: two-stage least squares (IV-2SLS)
stata_equivalent:
  - ivregress 2sls
r_equivalent:
  - AER::ivreg
  - linearmodels IV2SLS
status: mature
tier: 1
references:
  - hausman1978
  - hansen1982
  - stockyogo2005
---

# IV/2SLS Instrumental Variables Regression in Python

> **Estimator summary**: open-econs implements two-stage least squares (IV-2SLS) via linearmodels, supporting a three-part formula syntax (`y ~ exog | endog ~ instruments`), robust and unadjusted variance estimation, first-stage F diagnostics, Cragg-Donald weak-instrument statistics, and Hansen J overidentification tests.

## Overview

IV-2SLS estimates causal effects when one or more regressors are correlated with the error term (endogenous). open-econs wraps the `linearmodels.iv.IV2SLS` estimator, using the formulaic library for model-matrix construction and formula parsing.

Two code paths exist inside `iv()`:

- **New syntax** (`y ~ exog | endog ~ instruments`): explicitly separates exogenous controls from endogenous regressors. This is the recommended form.
- **Legacy syntax** (`y ~ rhs | instruments`): treats all right-hand-side variables as endogenous and all instrument-list variables as instruments. Emits a `FutureWarning` and is not recommended for applied work.

### Default Covariance

The default `cov_type="robust"` produces heteroskedasticity-consistent standard errors (HC0 equivalent without finite-sample correction). This differs from the open-econs library-wide HC2 default used by `oe.ols()` and `oe.fe()`.

## Mathematical Formulation

### Structural Model

\[
y_i = X_i \beta + W_i \gamma + \varepsilon_i, \quad i = 1, \dots, n
\]

where `y_i` is the scalar outcome, `X_i` is a `1 × k` row vector of **endogenous** regressors (correlated with `ε_i`), and `W_i` is a `1 × p` row vector of **exogenous** regressors (uncorrelated with `ε_i`, typically including a constant). OLS on this equation is inconsistent because `Cov(X, ε) ≠ 0`.

### First-Stage Equation

\[
X_i = Z_i \pi + W_i \delta + \nu_i
\]

where `Z_i` is a `1 × L` row vector of **instruments** (excluded from the structural equation). The first stage projects each endogenous regressor onto the instruments and exogenous controls. Instrument relevance requires `rank(π) = k` (the instruments are correlated with the endogenous regressors after partialling out `W_i`).

### Two-Stage Least Squares

1. Regress each endogenous variable `X_j` on `Z` and `W`. Obtain fitted values `\hat{X}_j = Z \hat{\pi}_j + W \hat{\delta}_j`.
2. Stack the fitted values: `\hat{X} = [\hat{X}_1, \dots, \hat{X}_k]`.
3. Regress `y` on `\hat{X}` and `W`:

\[
\hat{\beta}_{2SLS} = (\tilde{X}'\tilde{X})^{-1} \tilde{X}'y, \quad \tilde{X} = [\hat{X}, W]
\]

The 2SLS estimator is numerically equivalent to:

\[
\hat{\beta}_{2SLS} = (X' Z (Z' Z)^{-1} Z' X)^{-1} X' Z (Z' Z)^{-1} Z' y
\]

This is the standard IV estimator when `L ≥ k` (order condition for identification).

### Key Quantities of Interest

- Coefficients `β̂` and `γ̂` — structural parameters
- Standard errors `SE(β̂_j)` — type depends on `cov_type`
- z-statistics `z_j = β̂_j / SE(β̂_j)` (normal approximation)
- p-values (normal-based)
- 95% confidence intervals (normal-based)
- First-stage F-statistic per endogenous regressor
- Cragg-Donald Wald F-statistic (minimum of first-stage F values)
- Hansen J-statistic (overidentification test for `L > k`)

## Assumptions

1. **Linear structural equation**: The conditional mean is linear in parameters.
2. **Exclusion restriction**: `Cov(Z_i, ε_i) = 0` — instruments affect the outcome only through the endogenous regressors.
3. **Instrument relevance**: `rank(π) = k` — instruments are correlated with the endogenous regressors after controlling for `W_i`. Weak relevance (small first-stage F) leads to bias and non-normal inference.
4. **No perfect collinearity**: The projection `\tilde{X} = [\hat{X}, W]` has full column rank.
5. **Conditional homoskedasticity** (for `cov_type="nonrobust"` inference): `Var[ε_i | Z_i, W_i] = σ²`.
6. **Stable unit treatment value (SUTVA)**: No interference between units.

Assumptions 1–4 are required for consistency of 2SLS. Assumption 5 is only required for the classical ("unadjusted") variance estimator. Assumption 2 is untestable; assumption 3 can be assessed via the first-stage F-statistic.

## Estimator Derivation

2SLS solves the sample analogue of the population moment condition `E[Z_i ε_i] = 0`:

\[
\frac{1}{n} \sum_i Z_i (y_i - X_i \beta - W_i \gamma) = 0
\]

Substituting `ε_i = y_i - X_i β - W_i γ` and solving for the parameters gives the closed-form expression above. When `L = k` (just-identified), 2SLS is identical to the indirect least-squares estimator. When `L > k` (overidentified), 2SLS chooses the linear combination of instruments that minimises the asymptotic variance.

## Inference

### Covariance Estimators

The `cov_type` parameter is mapped to linearmodels conventions:

| open-econs `cov_type` | linearmodels `cov_type` | `debiased` | Description |
|------------------------|------------------------|------------|-------------|
| `"nonrobust"` | `"unadjusted"` | `False` | Homoskedastic `σ̂² (X̂'X̂)⁻¹` |
| `"robust"` | `"robust"` | `False` | Heteroskedastic-robust, **default**, HC0 equivalent |
| `"HC0"` | `"robust"` | `False` | HC0 alias (identical to `"robust"`) |
| `"HC1"` | `"robust"` | `True` | Small-sample corrected `n/(n−k)`, matches Stata `ivregress 2sls, robust` |
| `"HC2"` | `"robust"` | `False` | HC2 alias (identical to `"robust"`; no leverage adjustment for IV) |
| `"HC3"` | `"robust"` | `False` | HC3 alias (identical to `"robust"`; no jackknife adjustment for IV) |

**Key distinction**: HC0, HC2, and HC3 all map to the same linearmodels robust covariance because linearmodels does not implement leverage-adjusted HC2/HC3 for IV. The HC1 path applies the small-sample `n/(n−k)` correction via linearmodels' `debiased=True`. This differs from the OLS implementation, where HC0–HC3 are distinct.

The robust IV covariance is:

\[
V_{\text{robust}} = (\tilde{X}'\tilde{X})^{-1} \tilde{X}' \operatorname{diag}(\hat{e}_i^2) \tilde{X} (\tilde{X}'\tilde{X})^{-1}
\]

where `\hat{e}_i = y_i - [X_i, W_i]' \hat{\theta}` are the 2SLS structural residuals and `\tilde{X} = [\hat{X}, W]` are the second-stage regressors.

The unadjusted (homoskedastic) covariance is:

\[
V_{\text{nonrobust}} = \hat{\sigma}^2 (\tilde{X}'\tilde{X})^{-1}, \quad \hat{\sigma}^2 = \frac{1}{n - k - p} \sum_i \hat{e}_i^2
\]

### Default Behavior

| Parameter | Default |
|-----------|---------|
| `cov_type` | `"robust"` |
| Inference | z-based (normal approximation) |
| CIs | `β̂ ± 1.96 × SE` (normal) |

The default `"robust"` corresponds to HC0 without finite-sample correction. Use `cov_type="HC1"` to match Stata's `ivregress 2sls, robust` (which applies `n/(n−k)` correction). Use `cov_type="nonrobust"` for classical homoskedastic inference.

### Technical Deviations from External Software

| Feature | open-econs | Stata | R (`AER::ivreg`) |
|---------|------------|-------|-------------------|
| Default robust SE | HC0 (`robust`, no correction) | HC1 (`ivregress 2sls, robust`) | HC1 (default `vcovHC`) |
| HC0/HC2/HC3 distinction | All map to same linearmodels robust | Distinct formulas | Distinct formulas via `vcovHC` |
| Inference distribution | z-based (normal) | t-based (`df = n−k−p`) | t-based |
| Stata `ivregress 2sls, robust` match | `cov_type="HC1"` | Default robust | — |
| First-stage F | Per-endog variable; Cragg-Donald = min(F) | Same | Same |
| Weak instrument critical values | Not implemented (user compares against Stock-Yogo) | `estat weakiv` | Available |
| Hansen J overidentification | Yes (via linearmodels sargan) | `estat overid` | `summary()` output |
| Cluster-robust SE | Not available via `oe.iv()` API | `ivregress 2sls, vce(cluster)` | `vcovCL` |
| HAC SE | Not available | `ivregress 2sls, vce(hac)` | Not directly |
| Predict | Not implemented | Yes | `predict()` |

## Implementation Details

### Formula Interface

IV uses a three-part formula parsed by the custom `_parse_iv_formula` function (shared with `gmm()`):

```
y ~ exogenous_vars | endogenous_vars ~ instruments
```

For example:

```
y ~ w1 + w2 | x1 ~ z1 + z2
```

where:
- `y` — dependent variable (left of `~`)
- `w1 + w2` — exogenous regressors (between `~` and `|`)
- `x1` — endogenous regressor (between `|` and `~`)
- `z1 + z2` — instruments (right of the inner `~`)

When no inner `~` is present (legacy syntax):

```
y ~ x1 + w1 | z1 + z2
```

the implementation treats **all** right-hand-side variables (`x1 + w1`) as endogenous and `z1 + z2` as instruments. A `FutureWarning` is emitted.

The formula is parsed by the formulaic library. Missing columns raise a descriptive error.

### Backend

The estimation is delegated entirely to `linearmodels.iv.IV2SLS`:

1. `_parse_iv_formula` splits the formula, builds model matrices, aligns rows across y/X/instruments, and separates exogenous/endogenous indices.
2. `linearmodels.iv.IV2SLS(y, X_exog, X_endog, Z).fit(cov_type, debiased)` performs the 2SLS estimation.
3. Covariance mapping converts open-econs `cov_type` values to linearmodels conventions.
4. Results are extracted and wrapped in an `IVResult` object.

### Result Object

Returns an `IVResult` (immutable via `BaseModel._freeze()`):

| Attribute | Type | Description |
|-----------|------|-------------|
| `.coefficients` | `pd.Series` | Coefficient estimates, named by variable (intercept + exog + endog) |
| `.std_errors` | `pd.Series` | Standard errors |
| `.z_stats` | `pd.Series` | z-statistics (normal-based) |
| `.p_values` | `pd.Series` | p-values (normal-based) |
| `.conf_int` | `pd.DataFrame` | 95% confidence intervals (normal critical values) |
| `.rsd` | `float` | Residual standard deviation `√(σ̂²)` |
| `.nobs` | `int` | Observations used |
| `.df_resid` | `int` | Residual degrees of freedom |
| `.df_model` | `int` | Model degrees of freedom |
| `.cov_type` | `str` | Label for the covariance estimator used |
| `.first_stage_f` | `pd.Series` | First-stage F-statistic per endogenous variable |
| `.cragg_donald_stat` | `float` | Cragg-Donald Wald F-statistic (min of first-stage F) |
| `.hansen_j_stat` | `float` | Hansen J overidentification chi-squared statistic |
| `.hansen_j_p_value` | `float` | Hansen J p-value |
| `.fitted_values` | `pd.Series` | Fitted values `\hat{y} = X\hat{β} + W\hat{γ}` |
| `.residuals` | `pd.Series` | 2SLS structural residuals |
| `.call` | `dict` | Captured call arguments |

| Method | Description |
|--------|-------------|
| `.tidy()` | Coefficient table (DataFrame) with Variable, Coef, Std Err, z, P>|z|, 0.025, 0.975 |
| `.summary()` | Pretty-printed results including first-stage F, Cragg-Donald, Hansen J |
| `.vcov()` | Variance-covariance matrix (from linearmodels `_fit.cov`) |
| `.first_stage()` | DataFrame of first-stage F-statistics |
| `.export(path)` | Save as `.json` or `.csv` |
| `.to_latex()` / `.to_html()` | Export formatted tables |

`.predict()` is **not** implemented for IV results.

### Weak Instrument Diagnostics

First-stage F-statistics are extracted per endogenous variable from linearmodels' `first_stage.individual[var].f_statistic.stat`:

```python
for en_name in fitted.model.endog.cols:
    fs = fitted.first_stage
    if fs is not None and en_name in fs.individual:
        ind_res = fs.individual[en_name]
        f_stat = ind_res.f_statistic.stat
```

The **Cragg-Donald Wald F-statistic** is computed as the minimum of the per-endogenous-variable first-stage F-statistics:

```python
cragg_donald = min(fs_f_stats.values())
```

This is a standard weak-instrument diagnostic. The conventional Stock & Yogo (2005) critical values for the Cragg-Donald statistic depend on:
- the number of endogenous regressors (`k`)
- the number of instruments (`L`)
- the desired maximal bias or maximal size distortion

open-econs reports the Cragg-Donald statistic but does **not** embed critical values or produce a weak-instrument decision rule. Users should compare `result.cragg_donald_stat` against the Stock-Yogo tables.

### Overidentification Test

When `L > k` (more instruments than endogenous regressors), the Hansen J statistic is computed by linearmodels' `sargan` property:

```python
overid = fitted.sargan
hansen_j = float(overid.stat)
hansen_p = float(overid.pval)
```

The null hypothesis is that all instruments are valid (exogenous and correctly excluded from the structural equation). A small p-value indicates that one or more instruments may be invalid.

For just-identified models (`L = k`), the Hansen J statistic is NaN (the test is not defined).

### Covariance Label Convention

| `cov_type` value | `.cov_type` label |
|------------------|-------------------|
| `"nonrobust"` | `"nonrobust"` |
| `"robust"` | `"robust"` |
| `"HC0"` | `"HC0"` |
| `"HC1"` | `"HC1"` |
| `"HC2"` | `"HC2"` |
| `"HC3"` | `"HC3"` |

The label matches the user-facing `cov_type` (not the mapped linearmodels value).

### Covariance Mapping Details

The `_IV_COV_MAP` and `_IV_DEBIAS_MAP` control the translation:

```python
_IV_COV_MAP = {
    "nonrobust": "unadjusted",
    "HC0": "robust",
    "HC1": "robust",
    "HC2": "robust",
    "HC3": "robust",
    "robust": "robust",
    ...
}
_IV_DEBIAS_MAP = {
    "nonrobust": False,
    "HC0": False,
    "HC1": True,      # only HC1 activates small-sample correction
    "HC2": False,
    "HC3": False,
    "robust": False,
    ...
}
```

### Missing Data

Rows with any NaN in the formula variables are dropped with a `RuntimeWarning`. Missing values are handled by formulaic's `na_action="drop"` during model-matrix construction and by explicit index alignment between y/X and instrument matrices.

### Numerical Checks

- **Singular second-stage**: linearmodels raises an error if `\tilde{X} = [\hat{X}, W]` is rank-deficient.
- **Underidentification**: If `L < k`, linearmodels raises an error (order condition violated).
- **Weak identification**: No automatic warning is issued; users should inspect `cragg_donald_stat`.

## Stata / R Equivalents

### Stata

| open-econs | Stata | Notes |
|------------|-------|-------|
| `oe.iv("y ~ w | x ~ z", data=df)` | `ivregress 2sls y w (x = z)` | New syntax with exogenous controls |
| `oe.iv("y ~ w | x ~ z", data=df, cov_type="HC1")` | `ivregress 2sls y w (x = z), robust` | HC1 robust SEs matching Stata |
| `oe.iv("y ~ w | x ~ z", data=df, cov_type="nonrobust")` | `ivregress 2sls y w (x = z)` | Homoskedastic SEs (Stata default) |
| `r.first_stage_f`, `r.cragg_donald_stat` | `estat firststage` | First-stage diagnostics |

**Parameter mapping**: Stata's `ivregress 2sls` syntax places endogenous regressors in parentheses with instruments. open-econs uses a three-part formula with explicit `| endog ~ instruments` syntax. Stata defaults to homoskedastic SEs; open-econs defaults to robust (HC0). Use `cov_type="HC1"` to match Stata's `robust` option.

### R

| open-econs | R | Notes |
|------------|---|-------|
| `oe.iv("y ~ w | x ~ z", data=df)` | `AER::ivreg(y ~ w | x | z, data=df)` | `AER::ivreg` uses a different three-part syntax; default SEs differ |
| `oe.iv("y ~ w | x ~ z", data=df, cov_type="HC1")` | `AER::ivreg(y ~ w | x | z, data=df)` then `coeftest(vcov=vcovHC, type="HC1")` | Matches HC1 |
| `r.first_stage_f` | `summary(..., diagnostics=TRUE)` | R reports partial F | |

## API Examples

### Basic IV with Exogenous Controls (Recommended Syntax)

```python
import open_econs as oe

result = oe.iv("y ~ w | x ~ z1 + z2", data=df)
print(result.tidy())
#   Variable      Coef    Std Err        z    P>|z|      0.025      0.975
# 0   Intercept  1.023073  0.062101  16.4730  0.00000  0.901357  1.144789
# 1           w  2.105115  0.058751  35.8310  0.00000  1.989964  2.220265
# 2           x  0.486751  0.041963  11.5995  0.00000  0.404505  0.568998
```

### Legacy Syntax (All RHS Endogenous)

```python
result = oe.iv("y ~ x | z1 + z2", data=df)
```

This emits a `FutureWarning`.

### Homoskedastic Standard Errors (matching Stata default)

```python
result = oe.iv("y ~ w | x ~ z", data=df, cov_type="nonrobust")
```

### Robust Standard Errors (HC1 matching Stata `ivregress 2sls, robust`)

```python
result = oe.iv("y ~ w | x ~ z", data=df, cov_type="HC1")
```

### First-Stage Diagnostics

```python
print(result.first_stage())
#   Variable          F
# 0        x  42.772609

print(f"Cragg-Donald F: {result.cragg_donald_stat:.4f}")
# Cragg-Donald F: 42.7726

print(f"Hansen J: {result.hansen_j_stat:.4f} (p={result.hansen_j_p_value:.4f})")
# Hansen J: 1.7685 (p=0.4131)
```

### Export Results

```python
result.export("iv_results.json")
```

### Latex / HTML

```python
print(result.to_latex())
print(result.to_html())
```

### Stata Comparison

```python
# Stata: ivregress 2sls y w (x = z), robust
result = oe.iv("y ~ w | x ~ z", data=df, cov_type="HC1")
```

### Just-Identified Model

```python
result = oe.iv("y ~ w | x ~ z", data=df)
# Hansen J is NaN (just-identified: L = k = 1)
```

## Limitations

1. **No cluster-robust SEs**: The `cluster` parameter is not exposed in `oe.iv()`. For clustered IV, use linearmodels directly or preprocess manually.
2. **No HAC SEs**: Heteroskedasticity-and-autocorrelation consistent covariance is not available for IV.
3. **No HC2/HC3 distinction for IV**: Linearmodels does not implement leverage-adjusted HC2/HC3 for IV (HC0, HC2, HC3 all map to the same robust estimator). For leverage-corrected IV standard errors, use Stata or R.
4. **No `predict()`**: `IVResult` does not support `.predict()`. Fitted values are available via `.fitted_values`.
5. **No weak-instrument critical values**: The Cragg-Donald statistic is reported but not compared against Stock-Yogo critical values.
6. **No LIML**: Limited-information maximum likelihood is not implemented.
7. **No GMM-IV extensions**: Continuous-updating GMM, iterated GMM, or optimal-weight GMM for IV are not available (see `oe.gmm()` for linear GMM).
8. **No multiple endogenous equation systems**: Only a single structural equation is supported (no simultaneous-equation systems).
9. **No Context API**: `ctx.iv()` is not available; use `oe.iv()` directly with `data=df`.
10. **No weighted IV**: Weighted 2SLS is not supported.

## References

- @hausman1978
- @hansen1982
- @stockyogo2005
