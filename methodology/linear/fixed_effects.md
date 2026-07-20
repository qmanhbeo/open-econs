---
method: fe
aliases:
  - within estimator
  - entity fixed effects
  - two-way fixed effects
  - panel fixed effects
category: panel
api:
  - oe.fe()
context_api:
  - ctx.fe()
  - PanelContext.fe()
panel_api: []
panel_context_api: []
problem:
  - unobserved heterogeneity
  - panel regression
estimator: within transformation (one-way and two-way)
stata_equivalent:
  - xtreg, fe
  - reghdfe
r_equivalent:
  - plm(model="within")
  - fixest::feols
status: mature
tier: 1
references:
  - correia2017
  - mundlak1978
---

# Fixed Effects Panel Regression in Python — Stata `xtreg, fe` / R `plm::plm` Parity

> **Estimator summary**: open-econs implements one-way and two-way fixed-effects (within) estimators via group-mean subtraction for the one-way case and iterative alternating projections (Correia 2017) for the two-way case. The intercept is always absorbed by the fixed effects; standard errors are adjusted for the absorbed degrees of freedom set, matching Stata's `xtreg, fe` and `reghdfe` conventions.

## Overview

The fixed-effects (within) estimator controls for time-invariant unobserved heterogeneity by removing unit-specific (entity) means — and, optionally, time-specific means — before estimation. It answers the question: "holding fixed the unobserved time-invariant characteristics of each entity, what is the effect of X on Y?"

open-econs implements three distinct code paths inside `fe()`:

- **One-way entity FE** (`entity="id"` only): group-mean subtraction on y and each X column using a vectorised `pandas groupby().transform("mean")`, then OLS on the demeaned data.
- **One-way time FE** (`time="t"` only): same group-mean subtraction, grouped by the time column.
- **Two-way FE** (`entity="id"` and `time="t"`): iterative alternating projections (entity-demean → time-demean → repeat) until convergence at tolerance 1e-10, following the algorithm from Correia (2017) `reghdfe`.

After the within transformation and OLS fit, standard errors are rescaled by `sqrt( (n − k) / (n − n_absorbed − k) )` to account for the degrees of freedom consumed by the absorbed fixed effects. A scaled covariance matrix is stored on the result so that `.vcov()` returns values consistent with the reported standard errors.

## Mathematical Formulation

### One-Way Fixed Effects (Entity FE)

\[
y_{it} = \beta X_{it} + \alpha_i + \varepsilon_{it}, \quad i=1,\dots,N, \; t=1,\dots,T
\]

where `y_{it}` is the outcome, `X_{it}` is a `1 × k` row vector of time-varying regressors, `α_i` is an unobserved entity-specific intercept (the fixed effect), and `ε_{it}` is an idiosyncratic error term. Time-invariant regressors are collinear with `α_i` and are dropped mechanically by the within transformation.

The within transformation removes `α_i` by subtracting the entity-specific mean:

\[
y_{it} - \bar{y}_i = \beta (X_{it} - \bar{X}_i) + (\varepsilon_{it} - \bar{\varepsilon}_i)
\]

where `\bar{y}_i = T^{-1} \sum_{t} y_{it}` and similarly for `\bar{X}_i`. The transformed model has no intercept (the entity means are zero by construction). The within estimator is numerically identical to OLS on the demeaned data:

\[
\hat{\beta}_{\text{FE}} = \Big( \sum_i \sum_t (X_{it} - \bar{X}_i)' (X_{it} - \bar{X}_i) \Big)^{-1} \Big( \sum_i \sum_t (X_{it} - \bar{X}_i)' (y_{it} - \bar{y}_i) \Big)
\]

### Two-Way Fixed Effects

\[
y_{it} = \beta X_{it} + \alpha_i + \lambda_t + \varepsilon_{it}
\]

where `λ_t` is a time-specific intercept absorbing time-varying common shocks. The two-way within transformation removes both `α_i` and `λ_t`:

\[
y_{it} - \bar{y}_i - \bar{y}_t + \bar{y} = \beta (X_{it} - \bar{X}_i - \bar{X}_t + \bar{X}) + (\text{transformed error})
\]

For unbalanced panels this transformation cannot be expressed as a single group-mean subtraction; open-econs uses iterative alternating projections (Correia 2017) instead.

### Key Quantities of Interest

- Coefficients `β̂` — slopes on time-varying regressors only (intercept absorbed by FEs)
- Standard errors `SE(β̂_j)` — adjusted for absorbed degrees of freedom
- t-statistics `t_j = β̂_j / SE(β̂_j)` — using t-distribution with FE-adjusted `df_resid`
- p-values — using t-distribution with FE-adjusted `df_resid`
- 95% confidence intervals — using t-distribution critical values with FE-adjusted `df_resid`
- Within R² — share of within-entity (or within-time) variation explained
- Adjusted R², F-statistic, log-likelihood, AIC, BIC (all with FE-adjusted df)
- Residual standard deviation `rsd = √(SSR / df_resid_adj)`

## Assumptions

1. **Strict exogeneity**: `E[ε_{it} | X_i, α_i, λ_t] = 0` for all `i, t` (required for consistency of β̂).
2. **No perfect collinearity**: The within-transformed design matrix `Ẍ` has full column rank `k`. Time-invariant regressors are automatically collinear and are dropped.
3. **No autocorrelation / homoskedasticity** (required only for classical `cov_type="nonrobust"` inference; relaxed by HC or cluster-robust SEs).
4. **Independent errors across entities** (for cluster-robust SEs clustered at the entity level).
5. **Stable treatment effect**: `β` is constant across entities and time (homogeneous effect).

## Estimator Derivation

The FE estimator is derived from the least-squares minimisation problem including dummy variables for each entity:

\[
\min_{\beta, \alpha_1, \dots, \alpha_N} \sum_i \sum_t (y_{it} - X_{it}\beta - \alpha_i)^2
\]

The first-order condition for `α_i` gives `α̂_i = \bar{y}_i - \bar{X}_i \hat{\beta}`. Substituting back yields the within estimator. Equivalently, the Frisch-Waugh-Lovell theorem states that including entity dummies is identical to regression on group-demeaned variables.

## Inference

### Covariance Estimators

The FE estimator supports all covariance types that statsmodels provides on the within-transformed OLS fit:

| Estimator | Formula | Use Case |
|-----------|---------|----------|
| Classical (`nonrobust`) | `σ̂² (Ẍ'Ẍ)⁻¹` | Homoskedastic iid errors |
| HC0 | `(Ẍ'Ẍ)⁻¹ Ẍ' diag(eᵢ²) Ẍ (Ẍ'Ẍ)⁻¹` | Heteroskedastic-robust |
| HC1 | `(n/(n−k))·HC0` | Finite-sample corrected |
| HC2 | `(Ẍ'Ẍ)⁻¹ Ẍ' diag(eᵢ²/(1−hᵢᵢ)) Ẍ (Ẍ'Ẍ)⁻¹` | Leverage-adjusted |
| HC3 | `(Ẍ'Ẍ)⁻¹ Ẍ' diag(eᵢ²/(1−hᵢᵢ)²) Ẍ (Ẍ'Ẍ)⁻¹` | Conservative |
| Cluster | Sandwich clustered by `g∈{1,…,G}` | Correlated errors within groups |

**Degrees-of-freedom adjustment for standard errors**: All covariance types pass through the same post-fit SE rescaling:

\[
\text{SE}_{\text{adj}} = \text{SE}_{\text{statsmodels}} \times \sqrt{\frac{n - k}{n - n_{\text{absorbed}} - k}}
\]

where `n` is the number of observations, `k` is the number of regressors (excluding the absorbed effects), and `n_absorbed` is the number of absorbed FE groups: `N_entities` (one-way entity FE), `N_times` (one-way time FE), or `N_entities + N_times − 1` (two-way FE, subtracting 1 for the grand mean to avoid double-counting).

This scaling is exact for nonrobust and HC1 covariance (where the SE is proportional to `√(SSR / df)`). For HC2, HC3, and cluster-robust SEs it is an approximation that follows Stata's convention (`xtreg, fe` applies the same adjustment regardless of `vce()`).

**Confidence intervals and p-values**: re-computed using the FE-adjusted `df_resid` and Student's t-distribution, via `scipy.stats.t.ppf(0.975, df_resid_adj)`.

**Cluster-robust SEs**: When `cluster="col"` is specified, statsmodels' cluster-robust covariance is used (single clustering only; multi-way clustering is not supported for FE).

**Newey-West HAC SEs**: `cov_type="HAC"` with `lags` computes period-aggregation (Arellano / Driscoll-Kraay) heteroskedasticity- and autocorrelation-robust SEs. The score contributions `x_it · e_it` are summed *within each time period* across entities, then a Bartlett-kernel long-run variance is applied *across* periods. This requires `time` (which doubles as the time fixed-effects dimension, so HAC always incurs two-way FE) and `lags`. `hac_adjust=True` applies the `N/(N−K)` correction (Stata `newey` style); the default `False` is the original Newey & West (1987) formula. `cluster=` takes precedence over HAC. The same period-aggregation convention is shared with `ols()` and is validated against statsmodels `cov_nw_groupsum`.

### Default Behavior

| API entry point | Default `cov_type` |
|-----------------|-------------------|
| `oe.fe()`       | `"HC2"` |
| `ctx.fe()`      | `"HC1"` |
| `PanelContext.fe()` | `"HC1"` |

The `oe.fe()` top-level function matches the library-level HC2 default. The context methods (`ctx.fe()`, `PanelContext.fe()`) default to HC1 for historical consistency with earlier panel-context defaults; this is a documented discrepancy.

### Technical Deviations from External Software

| Feature | open-econs | Stata | R |
|---------|------------|-------|---|
| Default SE for `fe()` | HC2 (top-level) or HC1 (context) | `xtreg, fe` defaults to iid; `vce(robust)` uses HC1 | `plm(model="within")` defaults to iid |
| Cluster SE df adjustment | `√((n−k)/(n−n_absorbed−k))` applied to all SEs | Same | Different implementations |
| Two-way FE algorithm | Iterative alternating projections (Correia 2017) | `reghdfe` uses same algorithm | `fixest::feols` uses same algorithm |
| Absorbed df set | Entity + time − 1 | `reghdfe` counts all absorbed FE levels | `fixest` counts all absorbed FE levels |
| Within R² for two-way | SST uses entity-only demeaned y (Stata's `e(r2_w)`) | `xtreg` reports `r2_w` | `plm` reports within R² on two-way-demeaned data |
| Multi-way FE absorption | Entity + time only (two-way max) | `reghdfe` absorbs arbitrary high-d FEs | `fixest::feols` absorbs arbitrary high-d FEs |
| Weights | Not supported | `xtreg, fe` supports `[aw=pw]` | `plm` supports weights |
| Intercept in formula | Must not be included (absorbed by FE) | Absorbed automatically | Explicitly excluded |
| First-difference | Separate `diff()` method | `xtreg, fd` | `plm(model="fd")` |

## Implementation Details

### Formula Interface

Fixed effects do not appear in the formula string. The formula describes only the time-varying regressors:

```python
"y ~ x1 + x2"              # basic
"y ~ x1 + x2 - 1"          # no-intercept (redundant — intercept is absorbed)
"y ~ C(factor_var)"        # categorical encoding for regressors
```

Entity and time effects are specified via the `entity` and `time` keyword arguments:

```python
oe.fe("y ~ x1 + x2", data=df, entity="country", time="year")
```

Do not include entity or time dummy variables in the formula. The formula is parsed with [formulaic](https://github.com/matthewwardrop/formulaic); a missing-column error is raised with a list of available columns if formulaic fails.

### One-Way Demeaning Algorithm

The `_demean` function implements group-mean subtraction in O(n) time via `pandas groupby().transform("mean")`:

1. Reshape y to (n × 1) if 1-D.
2. Concatenate y (and each column of X) with the group labels.
3. Compute group means via `df.groupby("__g")["c0"].transform("mean")`.
4. Subtract group means from the original values.

This avoids forming the full `n × G` dummy matrix (O(nG) memory), which is the bottleneck for large panels.

### Two-Way Demeaning Algorithm (Iterative Alternating Projections)

The `_demean_two_way` function implements the algorithm from Correia (2017) `reghdfe`:

1. Start with y and X as provided.
2. Apply entity-demean via `_within_transform(y, entity_arr)`.
3. Apply time-demean via `_within_transform(y, time_arr)`.
4. Compare against the values before this iteration; if both y and X changed by less than `tol=1e-10`, stop.
5. Otherwise repeat, up to `max_iter=100`.

Because the entity and time projections are orthogonal only in balanced panels, the algorithm converges geometrically. For unbalanced panels, multiple iterations are required. In practice convergence is reached in 2–10 iterations for typical panel sizes; the 100-iteration limit is a safety bound.

After demeaning, the all-zero intercept column (if present in the original formula) is detected by its column name `"Intercept"` and dropped before fitting, so statsmodels computes the correct rank and degrees of freedom.

### Absorbed Fixed Effects Counting

The degrees of freedom consumed by the absorbed FE are:

| Case | `n_absorbed` |
|------|-------------|
| Entity FE only | `N_entities` |
| Time FE only | `N_times` |
| Two-way (entity + time) | `N_entities + N_times − 1` |

The subtraction of 1 in the two-way case avoids double-counting the grand mean (which is absorbed by both entity and time dummies). This matches Stata's `reghdfe` convention and is the standard correction for two-way FE models.

### Residual Degrees of Freedom

```python
df_resid_adj = max(n - n_absorbed - k, 1)
```

where `k` is the number of regressor coefficients (excluding the absorbed intercept).

### Standard Error Rescaling

After fitting statsmodels OLS with a given `cov_type`, standard errors are rescaled if the FE-adjusted df differs from statsmodels' df:

```python
df_old = fitted.df_resid  # = n - k (statsmodels does not know about FE)
if df_resid_adj != df_old and df_old > 0:
    scale = sqrt(df_old / df_resid_adj)
    se_arr *= scale
    t_arr = coef_arr / se_arr
    p_arr = 2 * t.sf(abs(t_arr), df_resid_adj)
    crit = t.ppf(0.975, df_resid_adj)
    conf_arr = [coef - crit*se, coef + crit*se]
```

The scaled covariance matrix is stored as a private `_cov` attribute so that `.vcov()` returns values consistent with the reported standard errors.

### Within R²

The within R² follows Stata's `e(r2_w)` convention:

```python
if entity is not None:
    y_for_r2 = demean(y, entity_groups)      # entity-only demeaned
elif time is not None:
    y_for_r2 = demean(y, time_groups)        # time-only demeaned

sst = sum((y_for_r2 - mean(y_for_r2)) ** 2)
ssr = sum(residuals ** 2)
r2 = 1 - ssr / sst
```

For two-way FE, the denominator uses **entity-only demeaned** y (not two-way demeaned). This matches Stata's `xtreg y x z i.time, fe` where `e(r2_w)` is the within-entity R². It differs from some R packages (`plm`) that compute R² on the two-way-demeaned y.

A safety fallback uses statsmodels' `fitted.rsquared` if the computed R² is NaN or outside [0, 1].

### Adjusted R²

```python
adj_r2 = 1 - (1 - r2) * (n - 1) / df_resid_adj
```

Uses the FE-adjusted residual degrees of freedom.

### Result Object

Returns an `OLSResult` (immutable). Key attributes and methods:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.coefficients` | `pd.Series` | Slopes on time-varying regressors only |
| `.std_errors` | `pd.Series` | FE df-adjusted standard errors |
| `.t_stats` | `pd.Series` | t-statistics (FE df-adjusted) |
| `.p_values` | `pd.Series` | p-values (FE df-adjusted) |
| `.conf_int` | `pd.DataFrame` | 95% CI (FE t-critical values) |
| `.r_squared` | `float` | Within R² (Stata `e(r2_w)` convention) |
| `.adj_r_squared` | `float` | Adjusted within R² (FE df) |
| `.df_resid` | `int` | `n − n_absorbed − k` |
| `.rsd` | `float` | `√(SSR / df_resid_adj)` |
| `.cov_type` | `str` | `"cluster(col)"` or the `cov_type` value |

| Method | Description |
|--------|-------------|
| `.tidy()` | Coefficient table as DataFrame |
| `.summary()` | Printed results with FE-corrected diagnostics |
| `.vcov()` | FE-df-scaled variance-covariance matrix |
| `.predict(newdata)` | In-sample prediction on within-transformed X |
| `.wald_test(r_matrix)` | Wald test |
| `.f_test(r_matrix)` | F-test |
| `.diagnostics()` | Jarque-Bera, Breusch-Pagan, Durbin-Watson, Ramsey RESET on the within-transformed residuals |

### Missing Data

Rows with NaN in any formula variable are dropped with a `RuntimeWarning`. If zero rows remain, an error is raised. Entity and time columns may contain NaN (they are looked up from the original `data` using the non-dropped index).

### Collinearity Check

The condition number is computed on the demeaned, intercept-free design matrix after fitting. There is no explicit BKW-style warning (unlike `ols.py`), but if the demeaned design matrix is singular, statsmodels will raise a `LinAlgError` during fitting.

### Diagnostics

Available via `result.diagnostics()`. Note that these diagnostics apply to the **within-transformed residuals**, which have zero entity means by construction. The Breusch-Pagan test and Durbin-Watson statistic may be less informative for FE residuals than for OLS residuals.

## Stata / R Equivalents

### Stata

| open-econs | Stata | Notes |
|------------|-------|-------|
| `oe.fe("y ~ x1 + x2", data=df, entity="id")` | `xtreg y x1 x2, fe` | One-way entity FE; HC2 default vs Stata iid default |
| `oe.fe("y ~ x1 + x2", data=df, entity="id", time="t")` | `xtreg y x1 x2 i.t, fe` | Two-way FE with time dummies |
| `oe.fe("y ~ x1 + x2", data=df, entity="id", time="t")` | `reghdfe y x1 x2, absorb(id t)` | Two-way FE via reghdfe |
| `oe.fe("y ~ x1 + x2", data=df, entity="id", cluster="id")` | `xtreg y x1 x2, fe vce(cluster id)` | Cluster-robust FE SEs |
| `oe.fe("y ~ x1 + x2", data=df, entity="id", cov_type="nonrobust")` | `xtreg y x1 x2, fe` | iid SEs matching Stata default |
| `oe.fe("y ~ x1 + x2", data=df, entity="id", cov_type="HC1")` | `xtreg y x1 x2, fe vce(robust)` | HC1 robust SEs matching Stata |
| `oe.fe("y ~ x1 + x2", data=df, entity="id", time="t", cov_type="HAC", lags=2)` | `xtscc y x1 x2, lag(2)` (Driscoll-Kraay) | Period-aggregation Newey-West; HAC requires `time` (doubles as time FE) |

**Parameter mapping**: Stata's `xtreg y x1 x2, fe` estimates entity FE only. For two-way, Stata users include `i.time` in the regressor list or use `reghdfe`. open-econs uses the unified `entity=`/`time=` keyword interface. The within R² (`e(r2_w)`) is numerically identical between Stata and open-econs for both one-way and two-way FE.

### R

| open-econs | R | Notes |
|------------|---|-------|
| `oe.fe("y ~ x1 + x2", data=df, entity="id")` | `plm(y ~ x1 + x2, data=pdf, model="within", effect="individual")` | — |
| `oe.fe("y ~ x1 + x2", data=df, entity="id", time="t")` | `plm(y ~ x1 + x2, data=pdf, model="within", effect="twoways")` | R² differs: plm uses two-way demeaned SST |
| `oe.fe("y ~ x1 + x2", data=df, entity="id", time="t")` | `fixest::feols(y ~ x1 + x2 | id + t, data=df)` | Same iterative algorithm, same R² convention |
| `oe.fe("y ~ x1 + x2", data=df, entity="id", time="t", cov_type="HAC", lags=2)` | `plm(y ~ x1 + x2, data=pdf, model="within", effect="twoways") %>% lmtest::coeftest(vcov=vcovNW(., lag=2))` | Period-aggregation Newey-West (Arellano) |

## API Examples

### One-Way Entity Fixed Effects

```python
import open_econs as oe

result = oe.fe("income ~ education + age", data=df, entity="province")
print(result.tidy())
#   Variable      Coef    Std Err        t    P>|t|      0.025      0.975
# 0   education  0.062408  0.005412  11.5279  0.00000  0.051798  0.073018
# 1         age  0.015609  0.000938  16.6378  0.00000  0.013770  0.017449
```

### Two-Way Fixed Effects

```python
result = oe.fe(
    "income ~ education + age",
    data=df,
    entity="province",
    time="year",
)
```

### Cluster-Robust Standard Errors

```python
result = oe.fe(
    "income ~ education + age",
    data=df,
    entity="province",
    cov_type="nonrobust",
    cluster="province",
)
```

### PanelContext API

```python
pc = oe.PanelContext(df, entity="province", time="year")
r1 = pc.fe("income ~ education + age")           # two-way, HC1 default
r2 = pc.fe("income ~ education + age",           # two-way, cluster SEs
           cluster="province")
r3 = pc.fe("income ~ education + age",           # entity FE only
           entity="province", time=None)
```

### Context API

```python
ctx = oe.Context(df)
r = ctx.fe("income ~ education + age", entity="province", time="year")
```

### Comparison with Stata xtreg

```python
# Stata:    xtreg income education age, fe vce(robust)
# open-econs equivalent:
result = oe.fe("income ~ education + age", data=df, entity="province", cov_type="HC1")
```

```python
# Stata:    reghdfe income education age, absorb(province year)
# open-econs equivalent:
result = oe.fe("income ~ education + age", data=df, entity="province", time="year")
```

## Limitations

1. **No multi-way clustering**: The `cluster` parameter accepts a single column name only. For multi-way clustered errors, use `oe.ols()` with explicit `cluster=["a", "b", ...]` on the demeaned data manually, or use `reghdfe`.
2. **No high-dimensional FE absorption**: Only entity + time (two-way) are supported. Absorbing three or more high-dimensional fixed effects (e.g. `reghdfe` with `absorb(firm#year)` or `fixest::feols` with `| id + t + id^t`) is not implemented.
3. **No weights**: Weighted least squares within FE is not supported.
4. **No instrumental variables within FE**: For endogenous regressors with fixed effects, use `oe.iv()` with manually created demeaned variables, or use a panel IV estimator.
5. **No random coefficients**: The FE model imposes `β` constant across entities and time.
6. **No automatic first-difference**: For FD estimation, use `PanelContext.diff()`.
7. **Context API HC1 default**: `ctx.fe()` and `PanelContext.fe()` default to `cov_type="HC1"`, not the library-wide `"HC2"` default used by `oe.fe()`.
8. **No unbalanced-panel R² for time-only FE**: The R² computation for one-way time FE follows the same within-R² formula as entity FE, using time-demeaned SST. This may differ from packages that compute overall R² for time-only models.
9. **No nonlinear FE**: Fixed-effects logit, probit, Poisson, or other nonlinear panel models are not implemented.

## References

- @correia2017
- @mundlak1978
