---
method: did_gardner
aliases:
  - DID2S
  - two-stage differences-in-differences
  - Gardner estimator
category: causal_inference
api:
  - oe.did_gardner()
context_api: []
problem:
  - heterogeneous treatment timing
  - treatment effect heterogeneity
  - staggered adoption
estimator: Gardner (2022) two-stage difference-in-differences (DID2S)
stata_equivalent: []
r_equivalent:
  - did2s::did2s
status: mature
tier: 1
references:
  - gardner2022
  - callawaysantanna2021
---

# Gardner (2022) Two-Stage Difference-in-Differences (`did_gardner`)

> **Estimator summary**: open-econs implements the Gardner (2022) two-stage
> difference-in-differences (DID2S) estimator. Stage 1 regresses the
> outcome on covariates / fixed effects using only **untreated** observations;
> stage 2 regresses the stage-1 residuals on the treatment indicator using
> **all** observations. The coefficient on the treatment indicator is the ATT.
> Standard errors are cluster-robust via two-stage influence functions,
> matching R's `did2s::did2s()`.

## Overview

Gardner (2022) proposes a two-stage procedure that cleanly separates the
estimation of untreated potential outcomes (stage 1) from the estimation of
the treatment effect (stage 2). This avoids the contamination of already-treated
units acting as controls that plagues two-way fixed-effects (TWFE)
regressions under staggered adoption.

The estimator:

1. **Stage 1**: Regress $Y$ on covariates and fixed effects using only
   **untreated** observations (never-treated and pre-treatment). This estimates
   the relationship between outcomes and covariates in the absence of treatment.
2. **Stage 2**: Regress the stage-1 residuals $\hat{u}_1$ on the treatment
   indicator using **all** observations. The coefficient on treatment is the ATT.
3. **Cluster-robust SEs** are computed via the full two-stage influence
   function, not a naive single-stage cluster VCE.

This is the estimator implemented by R's `did2s::did2s()` (v1.2.1,
non-bootstrap path). **No Stata anchor exists** for the Gardner DID2S
estimator; parity is established only against R `did2s`.

## Mathematical Formulation

### Two-Stage Estimator

Let $D_{it}$ be the binary treatment indicator and let the untreated subset be
$\mathcal{U} = \{(i,t) : D_{it} = 0\}$.

**Stage 1** (untreated only):

\[
\hat{\gamma}_1 = \arg\min_{\gamma_1} \sum_{(i,t) \in \mathcal{U}}
  \left( Y_{it} - X_{it}^{(1)\prime} \gamma_1 \right)^2
\]

where $X_{it}^{(1)}$ are the stage-1 regressors (covariates + fixed
effects). The stage-1 residuals are $\hat{u}_{1,it} = Y_{it} - X_{it}^{(1)\prime}\hat{\gamma}_1$.

**Stage 2** (all observations):

\[
\hat{\gamma}_2 = \arg\min_{\gamma_2} \sum_{i,t}
  \left( \hat{u}_{1,it} - D_{it}\,\alpha - X_{it}^{(2)\prime}\delta \right)^2
\]

The ATT is $\widehat{\text{ATT}} = \hat{\alpha}$, the coefficient on the treatment
indicator in stage 2.

### Two-Stage Influence Function

A naive single-stage cluster-robust VCE (second-stage residuals only)
underestimates the standard error because it ignores first-stage estimation
uncertainty. The correct two-stage influence function is:

\[
IF_i = IF_{i}^{fs} - IF_{i}^{ss}
\]

\[
IF_{i}^{ss} = (X_2'X_2)^{-1} X_2'\, \tilde{u}_{2,i}
\qquad\text{(second-stage OLS IF)}
\]

\[
IF_{i}^{fs} = (X_2'X_2)^{-1} \hat{\gamma}' X_{10}'\, \tilde{u}_{1,i}
\qquad\text{(first-stage IF)}
\]

where:

- $X_2$ is the stage-2 design matrix (all observations).
- $X_{10}$ is $X_1$ (the stage-1 design matrix, **all** observations)
  with the treated rows zeroed out (matching R `did2s`, which uses the
  original $X_1$ on the right-hand side of the cross-regression, **not**
  the zeroed-out $X_{10}$).
- $\hat{\gamma} = (X_{10}'X_{10})^{-1} (X_1'X_2)$ is the cross-regression
  coefficient.
- $\tilde{u}_{1,i}$, $\tilde{u}_{2,i}$ are the per-entity cluster sums of the
  first- and second-stage residuals.

The cluster-robust VCE is then $V = (X_2'X_2)^{-1} \left(\sum_c \hat{u}_c \hat{u}_c'\right) (X_2'X_2)^{-1}$
built from the two-stage influence functions, where $c$ indexes clusters.

### Identification Assumptions

Gardner (2022) requires:

1. **Staggered adoption (irreversibility)**: The treatment indicator is weakly
   increasing in $t$ for each unit.
2. **No anticipation**: Treatment has no effect before the treatment period.
3. **Parallel trends conditional on covariates**: Conditional on $X$, the
   average untreated outcome path is the same for treated and untreated units.
4. **First-stage linearity**: The untreated outcome is linear in the stage-1
   covariates / fixed effects.

## Inference

### Cluster-Robust Variance (Two-Stage IF)

Standard errors are cluster-robust, computed from the full two-stage influence
function above (not a single-stage VCE). This matches R `did2s::did2s()`
v1.2.1 (non-bootstrap path). A naive single-stage VCE underestimates
the SE by ~17% (it yields SE ≈ 0.4191 instead of the correct 0.5026).

### Default Behavior

| Setting | Default | Notes |
|---------|---------|-------|
| `first_stage` | required | RHS formula estimated on **untreated** observations only |
| `second_stage` | required | RHS formula estimated on **all** observations |
| `cluster` | `None` | If set, two-stage cluster-robust SEs; else iid SEs |
| `treatment` | required | Binary treatment indicator (1 = treated) |

### Technical Notes vs R `did2s::did2s()`

1. **Cross-regression uses original `X1`**: R's `did2s` computes
   `gamma = (X10'X10)^{-1} (X1'X2)` — the **original** `X1` (all
   observations) on the right side, not the zeroed-out `X10`. A naive
   implementation that zeroed treated rows in both sides of the cross-product
   produced SE = 0.4191 instead of the correct 0.5026. Source-confirmed
   by re-reading R's `did2s:::did2s()`.
2. **Two-stage IF**: The SE accounts for first-stage estimation uncertainty
   via the full $IF = IF^{fs} - IF^{ss}$; a second-stage-only VCE is
   incorrect.
3. **No Stata anchor**: Stata has no `did2s` equivalent with numerical
   parity; parity is established only against R `did2s`.

## Stata / R Equivalents

### R

| open-econs | R | Notes |
|------------|---|-------|
| `oe.did_gardner(data=df, y="y", first_stage="0 + factor(entity) + factor(time)", second_stage="treat", treatment="treat", cluster="entity")` | `did2s::did2s(y ~ x | entity + time, treat ~ 0 | entity + time, data=df, cluster=~entity)` | Both use two-stage IF SEs |

**Parameter mapping:**

| open-econs | R did2s | Notes |
|------------|---------|-------|
| `first_stage` | first-stage formula (untreated only) | Covariates + FE for untreated outcome |
| `second_stage` | second-stage formula | Typically the treatment indicator |
| `treatment` | `treat` | Binary treatment indicator |
| `cluster` | `cluster = ~entity` | Two-stage cluster-robust SEs |

### Stata

**No Stata equivalent.** The Gardner (2022) DID2S estimator is not
implemented in a standard Stata command with numerical parity. Do not treat
any Stata command as an equivalent anchor.

## API Examples

### Basic Gardner (2022) DID2S

```python
import open_econs as oe

result = oe.did_gardner(
    data=df,
    y="y",
    first_stage="0 + factor(entity) + factor(time)",
    second_stage="treat",
    treatment="treat",
    cluster="entity",
)
print(result.summary())
print(result.att)            # ATT (coefficient on treatment)
print(result.att_se)
```

### With Covariates

```python
result = oe.did_gardner(
    data=df,
    y="y",
    first_stage="0 + factor(entity) + factor(time) + x + z",
    second_stage="treat",
    treatment="treat",
    cluster="entity",
)
print(result.tidy())         # full coefficient table
print(result.vcov())        # variance-covariance matrix
```

### Accessing Results

```python
r = oe.did_gardner(
    data=df, y="y",
    first_stage="0 + factor(entity) + factor(time)",
    second_stage="treat",
    treatment="treat",
    cluster="entity",
)

# Key quantities
print(r.att)              # ATT estimate
print(r.att_se)           # standard error
print(r.att_t_stat)       # t-statistic
print(r.att_p_value)      # p-value

# Full output
print(r.tidy())           # coefficients, SEs, t, p, CI
print(r.summary())         # printable summary
```

## Limitations

1. **No Stata parity anchor**: The estimator is validated only against R
   `did2s::did2s()` v1.2.1. No Stata command provides numerical parity.
2. **No bootstrap SEs**: Standard errors are analytic two-stage influence
   functions; bootstrap is not available for `did_gardner()`.
3. **Requires explicit formulas**: Unlike `did_cs()` (which infers structure
   from column names), `did_gardner()` requires explicit `first_stage` and
   `second_stage` formula strings.
4. **Untreated subset required**: Stage 1 is estimated on untreated
   observations only; if no untreated observations exist, the estimator
   raises an error.
5. **iid SEs when `cluster=None`**: Without a cluster variable, the SEs
   are iid OLS SEs on the stage-2 system, which will understate uncertainty
   for correlated panels. Always pass `cluster=` for panel data.

## References

- Gardner, John. 2022. "Two-Stage Differences in Differences."
  arXiv:2207.05943. Working paper.
- Callaway, Brantly, and Sant'Anna, Pedro H. C. 2021. "Difference-in-
  Differences with Multiple Time Periods." *Journal of Econometrics*, 225(2),
  200–230.
