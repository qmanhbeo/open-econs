---
method: arellano_bond
aliases:
  - Arellano-Bond
  - difference GMM
  - dynamic panel GMM
  - AB GMM
category: linear
api:
  - oe.abond()
context_api:
  - ctx.abond()
  - PanelContext.abond()
problem:
  - dynamic panel bias
  - endogenous lagged dependent variables
  - unobserved individual effects in dynamic panels
estimator: Arellano-Bond (1991) difference GMM
stata_equivalent:
  - xtabond2
r_equivalent:
  - plm::pgmm
status: mature
tier: 1
references:
  - arellano_bond1991
  - arellano_bover1995
  - blundell_bond1998
  - windmeijer2005
  - roodman2009
---

# Arellano-Bond Dynamic Panel GMM (Difference GMM) in Python

> **Estimator summary**: open-econs implements the Arellano-Bond (1991) difference GMM estimator for dynamic panel models, supporting collapsed/non-collapsed instruments, one-step/two-step estimation, Windmeijer (2005) corrected standard errors, Hansen J overidentification tests, and Arellano-Bond AR(1)/AR(2) serial-correlation tests.

## Overview

Arellano-Bond GMM estimates a dynamic panel model where the lagged dependent variable appears as a regressor:

$$
y_{it} = \alpha y_{i,t-1} + x_{it}'\beta + \mu_i + \epsilon_{it}
$$

The presence of the individual fixed effect $\mu_i$ creates the classic **Nickell bias**: the demeaned lagged dependent variable is correlated with the demeaned error term because both contain overlapping observations of $\epsilon$. This bias shrinks with T but does not disappear in short panels (Nickell 1981).

open-econs implements first-difference GMM (the original Arellano-Bond 1991 estimator), removing $\mu_i$ by first-differencing and using deeper lags of the dependent variable and predetermined regressors as GMM instruments. The estimator wraps a shared GMM core (`_gmm_core.estimate_gmm`) with the Arellano-Bond-specific first-difference weighting matrix $H = M'M$ and a `sig2_scale = 0.5` normalization.

**System GMM** (Arellano-Bover 1995 / Blundell-Bond 1998), which adds level equations with lagged differences as instruments, is **not** implemented.

## Mathematical Formulation

### Dynamic Panel Model

Consider a balanced or unbalanced panel of $N$ entities observed over $T_i$ periods:

$$
y_{it} = \sum_{s=1}^{L} \alpha_s y_{i,t-s} + x_{it}'\beta + \mu_i + \epsilon_{it}, \quad t = 1, \dots, T_i
$$

where:
- $y_{it}$ is the scalar outcome
- $\alpha_s$ are autoregressive coefficients (up to `lags=L` lags)
- $x_{it}$ is a $k \times 1$ vector of time-varying regressors
- $\mu_i$ is the unobserved individual fixed effect
- $\epsilon_{it}$ is the idiosyncratic error, assumed serially uncorrelated under the null

Standard within-group estimation (entity demeaning) is inconsistent because:

$$
\text{Cov}(y_{i,t-1} - \bar{y}_i, \epsilon_{it} - \bar{\epsilon}_i) \neq 0
$$

This is the **Nickell bias**: the within transformation creates a mechanical correlation between the transformed lagged dependent variable and the transformed error. The bias is $O(1/T)$ and can be severe for small T.

### First-Difference Transformation

open-econs removes $\mu_i$ via first-differencing rather than within-demeaning:

$$
\Delta y_{it} = \sum_{s=1}^{L} \alpha_s \Delta y_{i,t-s} + \Delta x_{it}'\beta + \Delta \epsilon_{it}, \quad t = L+2, \dots, T_i
$$

where $\Delta y_{it} = y_{it} - y_{i,t-1}$.

The first difference eliminates $\mu_i$ but introduces a new endogeneity problem: $\Delta y_{i,t-1} = y_{i,t-1} - y_{i,t-2}$ is correlated with $\Delta \epsilon_{it} = \epsilon_{it} - \epsilon_{i,t-1}$ through the $\epsilon_{i,t-1}$ term present in both. OLS on the first-differenced equation is therefore also inconsistent.

### Moment Conditions (Arellano-Bond Instruments)

Arellano and Bond (1991) propose using deeper lags of the dependent variable as instruments for the differenced equation. Under the assumption that $\epsilon_{it}$ is serially uncorrelated:

$$
E[y_{i,t-s} \, \Delta\epsilon_{it}] = 0 \quad \text{for } s \ge 2, \; t = 3, \dots, T_i
$$

These moment conditions exploit the fact that $y_{i,t-2}$ and earlier lags are correlated with $\Delta y_{i,t-1}$ (instrument relevance) but uncorrelated with $\Delta \epsilon_{it}$ (instrument exogeneity), provided the $\epsilon_{it}$ are not serially correlated.

For predetermined regressors $x_{it}$ (regressors correlated with past errors but not future errors):

$$
E[x_{i,t-s} \, \Delta\epsilon_{it}] = 0 \quad \text{for } s \ge 1
$$

For strictly exogenous regressors, the full vector of differences is available:

$$
E[\Delta x_{it} \, \Delta\epsilon_{it}] = 0
$$

### GMM Estimator

Let $Y$ be the stacked vector of $\Delta y_{it}$ (usable equations only), $X$ the stacked regressor matrix of differenced lags and differenced $x$ variables, and $Z$ the stacked instrument matrix. The GMM estimator minimises:

$$
\hat{\theta} = \arg\min_\theta \; g(\theta)' W g(\theta), \quad g(\theta) = Z'(Y - X\theta)
$$

where $W$ is the weighting matrix and $g(\theta)$ are the sample moment conditions.

**One-step estimator**: $W = (Z'HZ)^{-1}$ where $H = M'M$ is the first-difference operator matrix (tridiagonal with 2 on the diagonal and -1 on the off-diagonals). This weighting is optimal under conditional homoskedasticity.

**Two-step estimator**: $W = \left( \sum_i Z_i' \hat{e}_{1i} \hat{e}_{1i}' Z_i \right)^{-1}$ where $\hat{e}_{1i}$ are the one-step residuals. The two-step estimator is asymptotically efficient under heteroskedasticity.

## Implementation Details

### One-Step Weighting Matrix ($H$)

The first-difference operator $M$ maps levels to differences: $\Delta y = M y$. For a single entity with $T_i$ periods, $M$ is $(T_i - 1) \times T_i$ with $M[j,j] = -1$, $M[j,j+1] = 1$, and the first row zeroed. The $H = M'M$ matrix is:

$$
H = \begin{pmatrix}
0 & 0 & 0 & \dots \\
0 & 2 & -1 & \dots \\
0 & -1 & 2 & \dots \\
\vdots & \vdots & \vdots & \ddots
\end{pmatrix}
$$

open-econs constructs the block-diagonal $H$ via `_build_h` (`abond.py:83-100`), returning the tridiagonal (diagonal, off-diagonal) representation. The leading unusable period $t=0$ has $H[0,0]=1$ but is excluded from the usable equation set ($j \ge \min_j$), so only the 2/-1 structure enters the moment conditions.

The one-step weighting is $A_1 = (Z'HZ)^{-1}$, implemented in `_estimate_gmm` (`_gmm_core.py:99-105`). The `abond()` function passes $H$ explicitly as the `W` argument to `_estimate_gmm`.

### Two-Step Weighting Matrix ($S$)

The per-entity moment outer product from one-step residuals $e_{1i}$:

$$
S = \sum_{i=1}^N Z_i' e_{1i} e_{1i}' Z_i
$$

This is the optimal weighting matrix under arbitrary heteroskedasticity (Hansen 1982). The two-step weight is $A_2 = S^{-1}$ (`_gmm_core.py:141-144`).

The same $S$ matrix also serves as the middle of the one-step robust sandwich (`V1robust` in `_gmm_core.py:139-141`); this mirrors xtabond2's Mata source (Mata 450-464).

### Standard Errors and Windmeijer Correction

| Configuration | Variance formula | Notes |
|---|---|---|
| One-step, `robust=False` | $\hat{\sigma}^2 (X'Z A_1 Z'X)^{-1}$ | Classical one-step; sig2 scaled by 0.5 |
| One-step, `robust=True` | $V_1 (Z'ZSZ'Z) V_1$ multiplied by small-sample factor | Sandwich from same $S$; $V_1 = (X'Z A_1 Z'X)^{-1}$ |
| Two-step, `robust=False` | $(X'Z A_2 Z'X)^{-1}$ | Classical two-step |
| Two-step, `robust=True` | $V_{2,\text{robust}}$ | **Windmeijer-corrected** |

The **Windmeijer (2005) correction** addresses the downward bias in two-step standard errors when the number of instruments is large. It adds a correction term $D$ that accounts for the variability of the two-step weighting matrix:

$$
V_{2,\text{robust}} = V_2 + D \cdot V_{1,\text{robust}} \cdot D' + 2 D V_2
$$

where $D = V_2 (X'Z A_2) \sum_i (Z_i' e_{1i} e_{1i}' Z_i) \cdot X'Z A_2 + \text{outer}(Z_i' e_{1i}, A_2 Z_i' X_i)$ per xtabond2's Mata 510-523.

The Windmeijer correction is **only** applied when `step="two-step"` and `robust=True`. When `robust=False` for two-step, the conventional two-step variance $V_2$ is used (which is known to be downward-biased in finite samples).

### Small-Sample Correction

Following xtabond2's Mata source (lines 562-565), open-econs always applies the small-sample multiplier for abond:

| Configuration | Multiplier |
|---|---|
| One-step, `robust=False` | $N_{\text{obs}} / (N_{\text{obs}} - k)$ |
| All other configurations | $((N_{\text{obs}} - 1) / (N_{\text{obs}} - k)) \cdot (N / (N - 1))$ |

The variance is scaled by the branch-specific multiplier. The sig2 estimate is always scaled by $N_{\text{obs}} / (N_{\text{obs}} - k)$.

The residual variance normalization is $\hat{\sigma}^2 = 0.5 \cdot e'e / N_{\text{obs}}$ (the `sig2_scale=0.5` parameter), which reflects the fact that the first-differenced variance is $2\sigma^2_\epsilon$ under homoskedasticity.

### Instrument Construction

open-econs builds instruments using two distinct code paths chosen by the `collapse` parameter (default `True`).

#### Collapsed Instruments (Default)

Collapsing follows Roodman (2009): for each lag depth $d$, one instrument column is created per GMM base variable. The instrument for lag depth $d$ for the dependent variable at usable equation period $j$ is:

$$
z_{j,d} = y_{i, j - L - d}
$$

where $L$ is the number of lags of $y$ included in the model. This is the collapsed equivalent of the staircase matrix: each column averages across all time periods, yielding $|\text{depths}|$ instruments per GMM variable rather than $\sum_i (T_i - d)$.

The instrument count in collapsed mode (`abond.py:341`):

$$
L = |D| \cdot (1 + |G|) + |I|
$$

where $|D|$ = number of valid depths, $|G|$ = number of predetermined (GMM-endogenous) regressors, $|I|$ = number of strictly exogenous regressors.

Degenerate depths (where $T - \max(\min_j, d) < 2$) are dropped in collapsed mode (`abond.py:315-324`), mirroring xtabond2's silent column-dropping behavior.

#### Non-Collapsed Instruments (Full Staircase)

When `collapse=False`, open-econs constructs the full block-diagonal "staircase" instrument matrix. For each entity $i$ and lag depth $d$, the instrument block for $y$ is a $(T_i - L - d) \times 1$ column with a single non-zero at position $j = d + L$ (`_build_noncollapsed_gmm_block`, `abond.py:12-40`).

The per-entity instrument column count is:

$$
L_i = \sum_{d \in \text{depths}} \max(0, T_i - d - L) + |G| \sum_{d \in \text{depths}} \max(0, T_i - d) + |I|
$$

Rows $j < \min_j$ are excluded from the estimation (usable equations only), but the full $T_i \times L_i$ matrix is constructed with row-level non-zero count assertions (`abond.py:427-436`) to ensure structural correctness.

#### Lag Depth Determination

Valid depths run from $2$ to $\max L$ where:

$$
\text{maxL} = \min(\text{max\_iv\_lag}, T_{\max} - 1)
$$

where $T_{\max}$ is the longest entity in the panel. When `max_iv_lag=None`, all available depths are used (defaults to $T_{\max} - 1$).

#### Strictly Exogenous Instruments

Regressors listed in the `exogenous` parameter (analogous to Stata's `iv()`) are instrumented with their own current-period differences:

$$
z = \Delta x_{it} = x_{it} - x_{i,t-1}
$$

These do not expand with depth — one column per exogenous regressor regardless of panel length.

### Diagnostic Tests

#### Hansen J Overidentification Test

When $L > p$ (more instruments than regressors), the Hansen J statistic tests the validity of the overidentifying restrictions:

$$
J = g(\hat{\theta})' A_{\text{used}} \, g(\hat{\theta}) \sim \chi^2_{L-p}
$$

where $A_{\text{used}}$ is $A_1$ (one-step) or $A_2$ (two-step). The test is implemented in `_gmm_core.py:206-210`. The null hypothesis is that all instruments are valid. The degrees of freedom are $L - p$.

For just-identified models ($L = p$), the J statistic is not defined and returned as NaN.

#### Arellano-Bond AR(1) and AR(2) Tests

The AR tests check for serial correlation in the first-differenced residuals, implemented in `_ar_test` (`abond.py:117-180`) mirroring xtabond2's Mata `_ARTests` (lines 1098-1167).

**AR(1)**: First-order serial correlation in $\Delta\epsilon_{it}$ is expected by construction (the difference $(\epsilon_{it} - \epsilon_{i,t-1})$ mechanically correlates with $(\epsilon_{i,t-1} - \epsilon_{i,t-2})$). A significant AR(1) test is normal and expected.

**AR(2)**: Second-order serial correlation in $\Delta\epsilon_{it}$ implies correlation between $\epsilon_{i,t-1}$ and $\epsilon_{i,t-2}$ in levels, which violates the Arellano-Bond moment conditions. A significant AR(2) test (p < 0.05) indicates that the instruments $y_{i,t-2}$ and deeper lags may be invalid.

The test statistic for lag $l$ (one-step non-robust):

$$
\text{AR}(l) = \frac{\sum_i e_i' w_{li}}{\sqrt{\sum_i w_{li}' H w_{li} \cdot \hat{\sigma}^2}}
$$

and for two-step/robust:

$$
\text{AR}(l) = \frac{\sum_i e_i' w_{li}}{\sqrt{\sum_i (e_i' w_{li})^2 + \text{correction}}}
$$

where $w_{li}$ is the lag-$l$ residual vector and the correction term involves $X_i' w_{li}$ and the pre-small-sample variance. The test is asymptotically standard normal.

Both AR(1) and AR(2) are always reported. The implementation reconstructs the full $T$-length residual per entity (with position 0 hard-zeroed), matching Stata's `_ARTests` conventions.

### Formula Interface

The `abond()` function uses a standard formulaic formula with entity and time identifiers:

```python
oe.abond("y ~ x1 + x2", data=df, entity="id", time="year")
```

The lagged dependent variable(s) are added automatically based on the `lags` parameter (default 1). The right-hand side lists all additional regressors.

Regressors are partitioned into:
- **GMM-endogenous** (default): all regressors not listed in `exogenous`. Instrumented with deeper lags (predetermined).
- **Strictly exogenous**: regressors listed in `exogenous=[...]`. Instrumented with current differenced values.

### Result Object

Returns an `ArellanoBondResult` (immutable via `BaseModel._freeze()`), defined in `panel_results.py:270-364`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.coefficients` | `pd.Series` | Coefficient estimates with names `L1.y`, `L2.y`, ..., `<x1>`, `<x2>`, ... |
| `.std_errors` | `pd.Series` | Standard errors (Windmeijer-corrected when `step="two-step"` and `robust=True`) |
| `.z_stats` | `pd.Series` | z-statistics $\hat{\beta} / \text{SE}$ (normal-based) |
| `.p_values` | `pd.Series` | p-values (normal-based) |
| `.conf_int` | `pd.DataFrame` | 95% confidence intervals (normal critical values 1.96) |
| `.step` | `str` | `"one-step"` or `"two-step"` |
| `.lags` | `int` | Number of lags of the dependent variable |
| `.n_entities` | `int` | Number of panel entities $N$ |
| `.n_obs` | `int` | Usable first-differenced equations |
| `.n_instruments` | `int` | Number of instrument columns $L$ |
| `.hansen_j` | `float` | Hansen J chi-squared statistic |
| `.hansen_j_pvalue` | `float` | Hansen J p-value |
| `.hansen_j_dof` | `int` | Hansen J degrees of freedom $L - p$ |
| `.sig2` | `float` | Residual variance estimate $\hat{\sigma}^2$ |
| `.ar1_stat` | `float` | AR(1) test z-statistic |
| `.ar1_pvalue` | `float` | AR(1) p-value |
| `.ar2_stat` | `float` | AR(2) test z-statistic |
| `.ar2_pvalue` | `float` | AR(2) p-value |
| `.call` | `dict` | Captured call arguments |

| Method | Description |
|--------|-------------|
| `.tidy()` | Coefficient table (DataFrame) with Variable, Coef, Std Err, z, P>|z|, 0.025, 0.975 |
| `.summary()` | Pretty-printed results with Hansen J and AR tests |
| `.to_dict()` | Dictionary form of results including step, lags, Hansen, AR tests |

`.predict()` is **not** implemented for `ArellanoBondResult`.

### Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `formula` | (required) | Two-sided formula, e.g. `"y ~ x1 + x2"` |
| `data` | (required) | `pd.DataFrame` |
| `entity` | (required) | Column name for panel entity index |
| `time` | (required) | Column name for panel time index |
| `lags` | `1` | Number of own lags of $y$ to include |
| `max_iv_lag` | `None` | Maximum lag depth for instruments (`None` = all available) |
| `step` | `"two-step"` | `"one-step"` or `"two-step"` |
| `exogenous` | `None` | List of strictly exogenous regressors |
| `collapse` | `True` | Collapse instruments (Roodman 2009) |
| `robust` | `False` | Use cluster-robust (Windmeijer-corrected for two-step) SEs |

### Default Behavior

| Parameter | Default | Notes |
|-----------|---------|-------|
| `step` | `"two-step"` | Efficient two-step GMM |
| `robust` | `False` | Classical GMM SEs for two-step (downward-biased); specify `robust=True` for Windmeijer correction |
| `collapse` | `True` | Reduced instrument count (recommended by Roodman 2009) |
| Inference | z-based (normal) | No small-sample t approximation |

### Panel Requirements

- Each entity must have at least **3 time periods** to form valid Arellano-Bond instruments ($T_i \ge 3$)
- Unbalanced panels are supported (each entity contributes $T_i - \min_j$ usable equations)
- The panel must be sorted by entity then time internally (handled by `np.lexsort` in `abond.py:277-280`)

### Missing Data

Rows with any NaN in the formula variables are dropped by formulaic's `na_action="drop"` during model-matrix construction.

### Numerical Checks

- **Insufficient time periods**: raises `ValueError` if any entity has fewer than 3 periods
- **Invalid step**: raises `ValueError` if `step` is not `"one-step"` or `"two-step"`
- **Invalid lags**: raises `ValueError` if `lags < 1`
- **No usable equations**: raises `ValueError` if no entity has usable equations after filtering
- **Singular weighting matrix**: falls back to pseudo-inverse in `_estimate_gmm`
- **Degenerate depths**: collapsed mode silently drops depths where too few rows would be non-zero (matching Stata behavior)

## Technical Deviations from Stata xtabond2

| Feature | open-econs | Stata `xtabond2` |
|---------|------------|-------------------|
| GMM type | Difference GMM only | Difference + system GMM |
| Default `collapse` | `True` | `False` (full instrument set) |
| Default `robust` | `False` | `True` (Windmeijer two-step) |
| `step` parameter | `"one-step"` / `"two-step"` | `onestep` option (default two-step) |
| `exogenous` syntax | `exogenous=["x1"]` | `iv(x1)` |
| GMM instrument variables | All non-exogenous regressors | `gmm()` option specifies variables explicitly |
| Multiple GMM groups | Not supported (single group) | `gmm(L.y, ...)` `gmm(x, ...)` with separate lag structures |
| AR tests | Always reported | `artests` option (default reported) |
| Small-sample correction | Always applied, following Mata 562-565 | `small` option and finite-sample corrections |
| `h()` bandwidth | Not supported | `h()` option for HAC weighting |
| `nested` option | Not supported | `nested` for iterated GMM starting values |
| `orthogonal` transform | Not implemented | `orthog` option (forward orthogonal deviations) |
| System GMM | Not implemented | `gmmstyle` / level equations |
| `artests` option | Not available (AR always reported) | `artests` / `noartests` |
| `sagan` option | Not available (Hansen always reported) | `sagan` / `nosa` |
| Constant instruments | Included as regressors | `iv()` with `_cons` |
| Inference distribution | z-based | z-based (same) |

## Stata / R Equivalents

### Stata

| open-econs | Stata | Notes |
|------------|-------|-------|
| `oe.abond("y ~ x", data=df, entity="id", time="t")` | `xtabond2 y L.y x, gmm(L.y) iv(x) robust twostep small` | Default differs: `collapse=True`, `robust=False` |
| `oe.abond("y ~ x", data=df, collapse=False, robust=True)` | `xtabond2 y L.y x, gmm(L.y) iv(x) robust twostep` | Matches xtabond2 defaults |
| `oe.abond("y ~ x", data=df, step="one-step")` | `xtabond2 y L.y x, gmm(L.y) iv(x) robust onestep` | One-step with robust SEs |
| `oe.abond("y ~ x", data=df, exogenous=["x"], robust=True)` | `xtabond2 y L.y x, gmm(L.y) iv(x) robust` | Treats x as strictly exogenous |
| `oe.abond("y ~ x", data=df, lags=2)` | `xtabond2 y L.y L2.y x, gmm(L.y) iv(x)` | Two lags of dependent variable |

**Parameter mapping**: Stata's `gmm()` specifies variables and their lag structure as instruments; open-econs automatically instruments the lagged dependent variable(s) and all non-exogenous regressors with deeper lags. Stata's `iv()` specifies strictly exogenous instruments; open-econs uses the `exogenous` parameter. Stata defaults to non-collapsed robust two-step; open-econs defaults to collapsed non-robust two-step. Specify `collapse=False, robust=True` to match xtabond2 defaults.

### R

| open-econs | R | Notes |
|------------|---|-------|
| `oe.abond("y ~ x", data=df, entity="id", time="t")` | `plm::pgmm(y ~ lag(y, 1) + x | lag(y, 2:99), data=df, effect="individual", model="twosteps", transformation="d")` | `plm::pgmm` has different defaults and requires explicit instrument specification |

## API Examples

### Default (Two-Step, Collapsed, Non-Robust)

```python
import open_econs as oe

result = oe.abond(
    "y ~ x",
    data=df,
    entity="entity",
    time="time",
)
print(result.summary())
#           Arellano-Bond Dynamic Panel (difference GMM, two-step)
# ======================================================================
# Dep. Variable:                      y
# No. Entities:                      120
# No. Observations:                  600
# No. Instruments (L):               10
# Lags of dep. var:                  1
# Hansen J:                   6.3659 (df=8, p=0.6061)
# AR(1) test:                -2.1179 (p=0.0342)
# AR(2) test:                 0.6547 (p=0.5127)
# ======================================================================
#     Variable      Coef    Std Err        z    P>|z|      0.025      0.975
# 0        L1.y  0.603057  0.041740  14.4472  0.00000  0.521248  0.684866
# 1           x  0.479378  0.069568   6.8907  0.00000  0.343028  0.615728
# ======================================================================
```

### Two-Step with Windmeijer Robust SEs (Matching xtabond2 Defaults)

```python
result = oe.abond(
    "y ~ x", data=df, entity="entity", time="time",
    collapse=False, robust=True,
)
```

### One-Step Estimation

```python
result = oe.abond(
    "y ~ x", data=df, entity="entity", time="time",
    step="one-step",
)
```

### Limiting Instrument Depth

```python
result = oe.abond(
    "y ~ x", data=df, entity="entity", time="time",
    max_iv_lag=4,
)
```

### Strictly Exogenous Regressors

```python
result = oe.abond(
    "y ~ x1 + x2", data=df, entity="entity", time="time",
    exogenous=["x2"],
)
# x2 is instrumented with its own current-period difference
# x1 and L.y are instrumented with deeper lags
```

### Context API

```python
ctx = oe.Context(df)
r1 = ctx.abond("y ~ x", entity="entity", time="time")

pc = oe.PanelContext(df, entity="entity", time="time")
r2 = pc.abond("y ~ x")
```

### Diagnostic Output

```python
print(f"Hansen J = {result.hansen_j:.4f} (p={result.hansen_j_pvalue:.4f})")
print(f"AR(1) = {result.ar1_stat:.4f} (p={result.ar1_pvalue:.4f})")
print(f"AR(2) = {result.ar2_stat:.4f} (p={result.ar2_pvalue:.4f})")
print(f"Number of instruments: {result.n_instruments}")
print(f"sig2 = {result.sig2:.6f}")
```

### Just-Identified Model

```python
# When L == p, Hansen J is returned as NaN (test is not defined)
# This can happen in small panels with few available depths
```

## Limitations

1. **Difference GMM only**: System GMM (Arellano-Bover 1995 / Blundell-Bond 1998) is not implemented. For persistent series where lagged levels are weak instruments, Stata's `xtabond2` with system GMM may perform better.

2. **No multiple GMM variable groups**: All endogenous regressors share the same lag depth structure. xtabond2 allows separate `gmm()` groups with independent lag specifications.

3. **No `h()` HAC weighting**: The `h()` bandwidth option for heteroskedasticity-and-autocorrelation consistent weighting in xtabond2 is not supported.

4. **No `nested` iterated GMM**: The `nested` option for iterated GMM starting from the one-step weights is not implemented.

5. **No `orthogonal` transformation**: Forward orthogonal deviations (Arellano-Bover 1995) are not available.

6. **No `artests` option**: AR tests are always reported and cannot be suppressed.

7. **No `sagan` option**: The Hansen J test is always reported.

8. **No `small` option**: The small-sample correction is always applied (matching xtabond2's `small` default).

9. **No `predict()`**: `ArellanoBondResult` does not support `.predict()`.

10. **No bootstrap SEs**: Standard errors are analytic only; bootstrap is not available.

## Stata / R Parity Status (2026-07-17)

- **Stata (`xtabond2`) — COMPLETE.** All 8 flavors (collapsed ×
  non-collapsed × one-step × two-step × robust × non-robust) are asserted to
  ≤1e-6 in `tests/stata/tests/test_stata_abond.py` (40 tests, green). Fixture
  `tests/stata/generate-fixtures/abond.do` + `tests/stata/fixtures/expected/abond.dta`
  (input `tests/stata/fixtures/inputs/df_panel.csv`, 30 entities × 5 periods).
  Stata uses `gmm(L.y, lag(2 4)) iv(x z) nolevel`, i.e. GMM instruments at
  lag depths 2–4; OE's `collapse=True` default with `max_iv_lag=4` reproduces
  this.
- **R (`plm::pgmm`) — BLOCKED.** `pgmm` is broken on R 4.6.1 (the dev R on this
  machine): it errors inside plm at `cbind(yX1[[i]], V1)` ("number of rows of
  matrices must match"), reproducible on the canonical `EmplUK` example for
  `effect="individual"` and `effect="twoways"`. plm 2.6.7 and 2.6.4 both fail;
  base `plm` (within) works. Root cause is a plm/R-4.6.1 incompatibility, not
  an OE bug. See FUTURE_WORK "ABOND R-Parity (BLOCKED)". When pgmm runs again,
  generate `tests/r/fixtures/expected/abond.json` from
  `tests/r/generate-fixtures/abond.R` using two-part formula
  `y | lag(y,-1)+lag(x,0)+lag(z,0) ~ lag(y,-2:-4)+lag(x,0)+lag(z,0)` with
  `effect="twoways", transformation="d"`, and add `tests/r/tests/test_r_abond.py`
  mirroring the 8 Stata flavors.

## References

- @arellano_bond1991
- @arellano_bover1995
- @blundell_bond1998
- @windmeijer2005
- @roodman2009
