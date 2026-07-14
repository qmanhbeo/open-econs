---
method: did_sa
aliases:
  - interaction-weighted DiD
  - event-study DiD
  - Sun-Abraham estimator
category: causal_inference
api:
  - oe.did_sa()
context_api: []
problem:
  - heterogeneous treatment timing
  - treatment effect heterogeneity over event time
  - staggered adoption
estimator: Sun & Abraham (2021) interaction-weighted event-study estimator
stata_equivalent: []
r_equivalent:
  - fixest::sunab
status: mature
tier: 1
references:
  - sunabraham2021
  - callawaysantanna2021
---

# Sun & Abraham (2021) Interaction-Weighted Difference-in-Differences (`did_sa`)

> **Estimator summary**: open-econs implements the Sun & Abraham (2021)
> interaction-weighted event-study estimator for staggered DiD settings. The
> estimator builds period × cohort interaction dummies, partials out entity and
> time fixed effects via Frisch-Waugh-Lovell (FWL), and runs OLS on the
> residualized system. The ATT is the cohort-weighted average of the
> period-level interaction coefficients. Standard errors are cluster-robust via
> the CRV1 sandwich with fixest's small-sample correction.

## Overview

Sun & Abraham (2021) propose an event-study estimator for staggered
treatment adoption that avoids the negative-weighting and already-treated-as-control
problems of two-way fixed effects (TWFE) regressions. Where Callaway &
Sant'Anna (2021) estimate a separate group-time ATT for every cohort × period
cell, Sun & Abraham estimate a single set of **relative-time** interaction
coefficients, one per event-time period (relative to each cohort's adoption
date).

The estimator:

1. Builds period × cohort interaction dummies from a calendar `period` and a
   `cohort` variable (the period in which each unit first becomes treated;
   `NA` for never-treated units).
2. Drops the reference event-time period (default `-1`) and the never-treated
   cohort.
3. Partialls out entity and time fixed effects via iterative demeaning
   (FWL), so the interaction coefficients are estimated on residualized data.
4. Runs OLS on the demeaned system (covariates + interaction dummies).
5. Detects collinearity among the demeaned interaction dummies only (using
   sequential Gram-Schmidt projection, matching fixest's Cholesky-based
   detection).
6. Computes the ATT as the cohort-weighted average of the period-level
   interaction coefficients, where weights are cohort-period cell shares.

This is the estimator implemented by R's `fixest::sunab()`. **No Stata
anchor exists** for the Sun-Abraham estimator (Stata's `eventstudyinteract`
is a distinct implementation; `csdid` targets Callaway-Sant'Anna instead).

## Mathematical Formulation

### Interaction-Weighted Estimator

Let $g_i$ denote unit $i$'s treatment cohort (the calendar period of first
treatment; $g_i = \infty$ if never treated) and let $e_{it} = t - g_i$ denote
$i$'s event-time in period $t$. For each relative period $k \neq -1$, define the
interaction indicator:

\[
D_{it}^{(g,k)} = \mathbf{1}\{g_i = g\} \cdot \mathbf{1}\{e_{it} = k\}
\]

The Sun-Abraham specification is:

\[
Y_{it} = \sum_{g} \sum_{k \neq -1} \beta_{g,k}\, D_{it}^{(g,k)} + \mu_i + \lambda_t + X_{it}'\gamma + \varepsilon_{it}
\]

where $\mu_i$ and $\lambda_t$ are entity and time fixed effects, and
$\beta_{g,k}$ is the cohort $g$ treatment effect at event-time $k$.

### ATT as Cohort-Weighted Aggregate

After estimating the full interaction system, the overall ATT is the
cohort-weighted average of the period-level aggregates:

\[
\text{ATT} = \sum_{k \neq -1} w_k \, \bar{\beta}_{\cdot,k},
\qquad
\bar{\beta}_{\cdot,k} = \sum_g w_{g,k}\, \beta_{g,k}
\]

where $w_{g,k}$ is the share of treated observations in cohort $g$ at event-time
$k$, and $w_k = \sum_g w_{g,k}$ normalizes across event-times. The ATT is
**not** the unweighted mean of all interaction coefficients; it is the
time::0 period-level aggregate (cohort-weighted).

### Identification Assumptions

Sun & Abraham (2021) require:

1. **Staggered adoption (irreversibility)**: Once treated, a unit stays treated.
   The binary treatment indicator is weakly increasing in $t$ for each unit.
2. **No anticipation**: Treatment has no effect on outcomes before the
   treatment period; the reference event-time $-1$ is assumed pre-treatment.
3. **Parallel trends conditional on covariates**: Conditional on $X$, the
   average outcome for each cohort would have followed the same path as the
   control group in the absence of treatment.
4. **Overlap**: For each cohort $g$ and event-time $k$, control units with
   similar covariate distributions exist.

## Inference

### Cluster-Robust Variance (CRV1 + fixest SSC)

Standard errors use the CRV1 sandwich estimator with fixest's default
small-sample correction (`ssc()`):

\[
\widehat{V} = (X'X)^{-1} \left( \sum_{c=1}^{G} X_c' \hat{u}_c \hat{u}_c' X_c \right) (X'X)^{-1}
\]

with the finite-sample scaling:

\[
\text{ssc} = \frac{G}{G-1} \cdot \frac{n-1}{n-K}
\]

where $G$ is the number of entity clusters, $n$ is the number of observations,
and $K = \text{nparams} - (G-1)$ is the degrees of freedom adjustment after
absorbing $G-1$ entity fixed effects. t-tests use $df = G - 1$.

The full estimator VCE is a $9 \times 9$ matrix (the 9 estimated
interaction coefficients); `SaDiDResult.vcov()` returns it as a
`pd.DataFrame`.

### Collinearity Detection

Collinearity among the demeaned interaction dummies is detected by sequential
projection (Gram-Schmidt in the original column order), matching fixest's
Cholesky-based detection. Dropped dummies are reported so the kept
coefficient index reconciles exactly with R `fixest::sunab()`'s `collvar`.

### Default Behavior

| Setting | Default | Notes |
|---------|---------|-------|
| `ref_period` | `-1` | Reference event-time dropped from interactions |
| `entity` / `time` | required for FE | Entity & time fixed effects partialled out via FWL |
| `cluster` | `None` | If set, CRV1 + SSC cluster-robust SEs; else iid SEs |
| `covariates` | `None` | Optional; absorbed alongside FE in the demeaning step |

### Technical Notes vs R `fixest::sunab()`

1. **SSC formula confirmed from source**: `G/(G-1) × (n-1)/(n-K)` with
   `K = nparams − (G−1)`, matching `fixest::vcov_cluster_internal` /
   `ssc_compute_K`.
2. **Collinearity detection** via sequential projection (Gram-Schmidt, original
   column order) matches fixest's Cholesky-based detection exactly.
3. **ATT definition**: the ATT is the time::0 period-level aggregate
   (cohort-weighted), not the mean of all interaction coefficients.
4. **No Stata anchor**: Stata's `eventstudyinteract` (Sun) and `csdid` are
   distinct packages with different defaults; parity is established only against
   R `fixest::sunab()` v0.14.2.

## Stata / R Equivalents

### R

| open-econs | R | Notes |
|------------|---|-------|
| `oe.did_sa(data=df, y="y", cohort="g", period="t", ref_period=-1, entity="id", time="t", cluster="id", covariates=["x"])` | `fixest::feols(y ~ x | id + t, sunab(g, t), cluster=~id)` | Both default to cluster-robust; ATT = cohort-weighted period aggregate |

**Parameter mapping:**

| open-econs | R fixest | Notes |
|------------|---------|-------|
| `cohort` | `sunab(cohort, ...)` | Cohort (first-treatment period); `NA` for never-treated |
| `period` | `sunab(..., period)` | Calendar time |
| `ref_period` | `ref` (default `-1`) | Reference event-time dropped |
| `entity` / `time` | `feols(... | id + t)` | Absorbed FE |
| `cluster` | `cluster = ~id` | CRV1 + SSC |
| `covariates` | RHS of `feols(y ~ x | ...)` | Covariates partialled out with FE |

### Stata

**No Stata equivalent.** The Sun-Abraham estimator is not implemented in a
standard Stata command with numerical parity. Stata's `eventstudyinteract`
(Sun) and `csdid` (Callaway-Sant'Anna) use different identification and
aggregation schemes; do not treat them as equivalent anchors.

## API Examples

### Basic Sun-Abraham DID

```python
import open_econs as oe

result = oe.did_sa(
    data=df,
    y="y",
    cohort="cohort",
    period="time",
    ref_period=-1,
    entity="entity",
    time="time",
    cluster="entity",
    covariates=["x"],
)
print(result.summary())
print(result.att)            # overall ATT
print(result.att_se)
```

### Period- and Cohort-Level Aggregates

```python
result = oe.did_sa(
    data=df,
    y="y",
    cohort="cohort",
    period="time",
    ref_period=-1,
    entity="entity",
    time="time",
    cluster="entity",
    covariates=["x"],
)

# Period-level aggregated coefficients (one per event-time period)
print(result.period_names)
print(result.period_coefs)
print(result.period_ses)

# Cohort-specific ATTs (time::0 coefficient per cohort, if non-collinear)
print(result.cohort_names)
print(result.cohort_coefs)
print(result.cohort_ses)
```

### Full Coefficient Vector

```python
r = oe.did_sa(
    data=df, y="y", cohort="cohort", period="time",
    ref_period=-1, entity="entity", time="time",
    cluster="entity", covariates=["x"],
)

# 9 interaction coefficients (raw) with SEs / t / p
print(r.tidy())

# Full 9x9 cluster-robust VCE
print(r.vcov())
```

## Limitations

1. **No Stata parity anchor**: The estimator is validated only against R
   `fixest::sunab()` v0.14.2. No Stata command provides numerical parity.
2. **No bootstrap SEs**: Standard errors are analytic CRV1 + SSC; bootstrap
   is not available for `did_sa()`.
3. **Single reference period**: Only a single `ref_period` (default `-1`) is
   supported; multiple-reference or normalized-to-zero schemes are not exposed.
4. **Collinearity handling is automatic**: Dropped interaction dummies are
   detected and excluded; the user cannot force-include a collinear dummy.
5. **Requires explicit `cohort` column**: Unlike `did_cs()` (which infers the
   cohort from a binary treatment indicator), `did_sa()` requires an explicit
   `cohort` variable with `NA` for never-treated units.

## References

- Sun, Liyang, and Abraham, Sarah. 2021. "Estimating Dynamic Treatment
  Effects in Event Studies With Heterogeneous Treatment Effects."
  *Journal of Econometrics*, 225(2), 175–199.
- Callaway, Brantly, and Sant'Anna, Pedro H. C. 2021. "Difference-in-
  Differences with Multiple Time Periods." *Journal of Econometrics*, 225(2),
  200–230.
