---
method: staggered_did
aliases:
  - Callaway-Sant'Anna DiD
  - CS2021 DiD
  - staggered difference-in-differences
  - heterogeneous treatment timing DiD
  - group-time ATT
category: causal_inference
api:
  - oe.staggered_did()
context_api: []
problem:
  - heterogeneous treatment timing
  - treatment effect heterogeneity
  - staggered adoption
estimator: Callaway-Sant'Anna (2021) group-time ATT with doubly-robust DiD (Sant'Anna & Zhao 2020)
stata_equivalent:
  - csdid
r_equivalent:
  - did
  - DRDID
status: mature
tier: 1
references:
  - callawaysantanna2021
  - santannazhao2020
  - abadie2005
---

# Callaway-Sant'Anna Staggered Difference-in-Differences in Python

> **Estimator summary**: open-econs implements the Callaway & Sant'Anna (2021) group-time average treatment effect estimator for staggered DiD settings, using the doubly-robust (DR) DiD approach of Sant'Anna & Zhao (2020) when covariates are present, and a simple outcome-regression 2×2 OLS interaction when they are not. The estimator handles heterogeneous treatment timing, provides influence-function-based cluster-robust standard errors, and supports never-treated or not-yet-treated control groups.

## Overview

Staggered difference-in-differences estimates the average treatment effect on the treated (ATT) in settings where units adopt treatment at different times (staggered adoption). The Callaway & Sant'Anna (2021) approach addresses a well-known problem with traditional two-way fixed effects (TWFE) DiD: when treatment timing is heterogeneous, TWFE regressions can produce biased estimates because already-treated units serve as controls for later-treated units.

open-econs implements the CS2021 group-time ATT estimator with two estimation methods:

- **`method="dripw"`** (default when covariates are provided): Doubly-robust DiD combining logit propensity score weighting with OLS outcome regression, following Sant'Anna & Zhao (2020) as implemented in the R `DRDID` package (`drdid_panel` / `trad` method).
- **`method="reg"`** (default when no covariates): Simple 2×2 OLS interaction (post × treated) with entity-clustered standard errors.

The estimator computes $ATT(g,t)$ for each cohort $g$ at each post-treatment period $t$, then aggregates to an overall ATT using equal-weighted influence-function aggregation. Standard errors are influence-function-based cluster-robust for `dripw`, entity-clustered OLS for `reg`.

## Mathematical Formulation

### Problem with Traditional Two-Way Fixed Effects DiD

The canonical TWFE DiD specification is:

\[
Y_{it} = \alpha_i + \lambda_t + \beta D_{it} + \epsilon_{it}
\]

where $\alpha_i$ are unit fixed effects, $\lambda_t$ are time fixed effects, and $D_{it}$ is the treatment indicator. When treatment timing is heterogeneous (staggered adoption), the TWFE estimator $\hat{\beta}$ is a weighted average of all possible 2×2 comparisons (Goodman-Bacon 2021). This creates two problems:

1. **Already-treated units as controls**: Units treated in earlier periods serve as controls for later-treated units, even though they are themselves under treatment. This contaminates the control group.
2. **Negative weighting**: When treatment effects vary across groups or over time, the TWFE weights can be negative, producing an ATT estimate that lies outside the convex hull of the individual group-time ATTs (de Chaisemartin & D'Haultfoeuille 2020; Sun & Abraham 2021; Callaway & Sant'Anna 2021).

The CS2021 estimator avoids these problems by constructing explicit group-time ATTs and aggregating them with non-negative weights.

### Group-Time Average Treatment Effects

Let there be $G$ cohorts indexed by treatment time $g \in \mathcal{G}$, and $T$ calendar periods $t \in \{1, \dots, T\}$. Each unit $i$ belongs to exactly one cohort $G_i = g$ (the period it first receives treatment) or is never treated ($G_i = \infty$). The potential outcomes are $Y_{it}(g)$ (treated from period $g$) and $Y_{it}(\infty)$ (never treated).

The **group-time average treatment effect** for cohort $g$ at time $t$ is:

\[
ATT(g,t) = E[Y_t(g) - Y_t(0) \mid G_g = 1]
\]

where $Y_t(g)$ is the outcome at time $t$ if treated from period $g$, $Y_t(0)$ is the outcome under never-treatment, and $G_g$ is the indicator for membership in cohort $g$. The $ATT(g,t)$ is defined only for post-treatment periods $t \geq g$.

### Identification Assumptions

Callaway & Sant'Anna (2021) require:

1. **Staggered adoption (irreversibility)**: Once a unit becomes treated, it remains treated in all subsequent periods. The treatment indicator $D_{it}$ is weakly increasing in $t$ for each unit $i$.
2. **No anticipation**: Treatment has no effect on outcomes before the treatment period. This is operationalized by using $g-1$ as the baseline period.
3. **Parallel trends based on comparison group**: Conditional on covariates $X$, the average outcome for the treated cohort would have followed the same path as the control group in the absence of treatment. The control group can be either never-treated units or not-yet-treated units.
4. **Overlap**: For each cohort $g$ and post-treatment period $t$, there exist control units with similar covariate distributions. This is enforced by trimming control units with propensity scores above 0.995.
5. **No anticipation**: Treatment has no causal effect before the treatment period. The baseline period $g-1$ is assumed to be unaffected by future treatment.

## Inference

### Influence Function Variance

The variance of each $ATT(g,t)$ is computed from the influence function representation. For each entity $i$ in the full sample, the cell-level RIF is:

\[
RIF_i(g,t) = \psi_i(g,t) + ATT(g,t)
\]

where $\psi_i(g,t)$ is the influence function for cell $(g,t)$. The shifted RIF ensures $\text{mean}(RIF) = ATT(g,t)$.

The cell-level variance is:

\[
V(ATT(g,t)) = \frac{1}{N^2} \sum_{i=1}^N (RIF_i(g,t) - \overline{RIF(g,t)})^2
\]

where $N$ is the total number of entities in the full sample (not just the cell sample). Entities not in the cell receive $RIF_i = 0$. There is **no small-sample correction** (no $N/(N-k)$ multiplier).

### Aggregated Variance

The overall ATT is the equal-weighted average across $K$ post-treatment cells:

\[
ATT = \frac{1}{K} \sum_{(g,t) \in \mathcal{P}} ATT(g,t)
\]

The aggregated influence function is:

\[
RIF^{agg}_i = \frac{1}{K} \sum_{(g,t) \in \mathcal{P}} RIF_i(g,t)
\]

The aggregated variance is:

\[
V^{agg} = \frac{1}{N^2} \sum_{i=1}^N (RIF^{agg}_i - \overline{RIF^{agg}})^2
\]

where $N$ is the total number of entities in the full sample. This matches the `did` R package's `aggte(type="simple")` and csdid's `saverif` + `csdid_stats simple` aggregation. It does **not** match Stata's `csdid_estat simple` command, which is known to be buggy (see caveat below).

### Bootstrap

When `bootstrap=True`, the overall ATT and its standard error are computed via entity-level resampling (500 replications by default). In each bootstrap replication:

1. Sample $N$ entities with replacement from the full sample.
2. Recompute all group-time ATTs on the bootstrap sample using the same method (`dripw` or `reg`).
3. Aggregate to an overall ATT using equal weighting.
4. After all replications, the bootstrap ATT is the mean of the bootstrap distribution, and the bootstrap SE is the sample standard deviation (with Bessel correction).

Bootstrap is **not** the default; analytic influence-function SEs are used when `bootstrap=False`.

### Default Behavior

| Setting | Default | Notes |
|---------|---------|-------|
| `method` | `"dripw"` if covariates provided, else `"reg"` | Auto-selected based on covariate presence |
| `cluster` | `entity` column | Entity-level clustering |
| `control_cohorts` | `"not_yet_treated"` | Uses both never-treated and not-yet-treated |
| `bootstrap` | `False` | Analytic IF SEs by default |
| `bootstrap_reps` | 500 | Only used when `bootstrap=True` |
| `seed` | `None` | No seed by default |

### Technical Deviations from Stata csdid

1. **`csdid_estat simple` is buggy**: Stata's `csdid_estat simple` command (csdid v1.6/v1.58) posts the raw per-(g,t) VCoV and prints element [1,1] — the first cell's SE — as the "simple" ATT SE. This is not an aggregation SE at all. open-econs matches the correct influence-function aggregation used by `csdid_stats simple` and the `did` R package's `aggte(type="simple")`. Users comparing against Stata should use `csdid_stats simple` or the `saverif(rif)` workflow, **not** `csdid_estat simple`.

2. **No `gvar` parameter**: Stata's `csdid` requires an explicit `gvar(treatment_year)` parameter. open-econs infers the treatment cohort from the binary treatment indicator by finding the first period in which each entity has `treatment=1`. This means entities that never have `treatment=1` are automatically treated as never-treated. Users migrating from Stata should ensure their treatment indicator is binary and correctly coded.

3. **No `ivar` parameter**: Stata's `csdid` requires `ivar(id)`. open-econs uses the `entity` parameter instead.

4. **No `time` parameter in Stata**: Stata's `csdid` uses the `time()` option. open-econs uses the `time` parameter.

5. **No `saverif` option**: Stata's `csdid` can save per-entity influence functions via `saverif(rif)`. open-econs does not expose the per-entity RIFs in the result object (they are used internally for variance computation).

6. **No `csdid_stats` equivalent**: Stata's `csdid_stats` command provides various aggregation schemes. open-econs provides only the "simple" equal-weighted aggregation.

7. **No `plot()` method**: The `StaggeredDiDResult` does not implement a `plot()` method. Users should extract the `event_study` DataFrame and plot manually.

8. **No `pretrends` test**: The CS2021 pre-trends test (based on pre-treatment periods) is not implemented.

9. **No `ggsynth` placebo test**: Placebo-based inference is not available.

## Stata / R Equivalents

### Stata

| open-econs | Stata | Notes |
|------------|-------|-------|
| `oe.staggered_did(df, y="y", entity="id", time="t", treatment="treat", covariates=["x","z"])` | `csdid y x z, ivar(id) time(t) gvar(treat)` | Both default to doubly-robust `dripw` |
| `oe.staggered_did(..., control_cohorts="never_treated")` | `csdid y x z, ivar(id) time(t) gvar(treat), never` | Restricts to never-treated controls |
| `oe.staggered_did(..., method="reg")` | `csdid y, ivar(id) time(t) gvar(treat), method=reg` | Outcome-regression only |

**Parameter mapping:**

| open-econs | Stata csdid | Notes |
|------------|-------------|-------|
| `entity` | `ivar()` | Panel entity identifier |
| `time` | `time()` | Time period variable |
| `treatment` | `gvar()` | open-econs infers cohort from binary treatment indicator; Stata requires explicit `gvar` |
| `covariates` | `xlist` | Covariates for the doubly-robust estimator |
| `method="dripw"` | `method(dripw)` | Default in both |
| `method="reg"` | `method(reg)` | Outcome-regression only |
| `control_cohorts="never_treated"` | `never` option | Restrict to never-treated controls |
| `control_cohorts="not_yet_treated"` | (default) | Uses not-yet-treated + never-treated |

### R

| open-econs | R | Notes |
|------------|---|-------|
| `oe.staggered_did(df, y="y", entity="id", time="t", treatment="treat", covariates=["x","z"])` | `did::att_gt(yname="y", tname="t", idname="id", gname="g", xformla=~x+z, data=df)` | Both default to doubly-robust |
| `oe.staggered_did(..., method="dripw")` | `DRDID::drdid_panel(y~x+z, t=t, treated=D, id=id, data=df, method="trad")` | Same DR estimator |
| `oe.staggered_did(..., control_cohorts="never_treated")` | `did::att_gt(..., control_group="never_treated")` | Restrict to never-treated controls |

**Parameter mapping:**

| open-econs | Stata csdid | R did | Notes |
|------------|-------------|-------|-------|
| `entity` | `ivar()` | `idname` | Panel entity identifier |
| `time` | `time()` | `tname` | Time period variable |
| `treatment` | `gvar()` | `gname` | open-econs infers cohort from binary treatment; Stata/R require explicit cohort variable |
| `covariates` | `xlist` | `xformla` | Covariates for DR estimator |
| `method="dripw"` | `method(dripw)` | `DRDID::drdid_panel(..., method="trad")` | Default doubly-robust |
| `method="reg"` | `method(reg)` | `did::att_gt(..., est_method="reg")` | Outcome-regression only |
| `control_cohorts="never_treated"` | `never` | `control_group="never_treated"` | Restrict to never-treated |
| `control_cohorts="not_yet_treated"` | (default) | `control_group="notyettreated"` | Default in all three |

## API Examples

### Basic Staggered DiD with Covariates

```python
import open_econs as oe

result = oe.staggered_did(
    data=df,
    y="y",
    entity="id",
    time="t",
    treatment="treat",
    covariates=["x", "z"],
)
print(result.summary())
print(result.att_group_time)
print(result.event_study)
```

### Staggered DiD without Covariates (Outcome Regression)

```python
result = oe.staggered_did(
    data=df,
    y="y",
    entity="id",
    time="t",
    treatment="treat",
    # No covariates -> method="reg" automatically
)
```

### Never-Treated Control Group Only

```python
result = oe.staggered_did(
    data=df,
    y="y",
    entity="id",
    time="t",
    treatment="treat",
    covariates=["x", "z"],
    control_cohorts="never_treated",
)
```

### Bootstrap Standard Errors

```python
result = oe.staggered_did(
    data=df,
    y="y",
    entity="id",
    time="t",
    treatment="treat",
    covariates=["x", "z"],
    bootstrap=True,
    bootstrap_reps=1000,
    seed=42,
)
```

### Accessing Group-Time Results

```python
# Group-time ATT table
print(result.att_group_time)

# Event-study path
print(result.event_study)

# Overall ATT
print(f"ATT = {result.att:.4f} (SE = {result.att_se:.4f}, p = {result.att_p:.3f})")

# Summary
print(result.summary())
```

## Limitations

1. **No system GMM**: Only difference GMM is implemented. System GMM (Arellano-Bover 1995 / Blundell-Bond 1998) is not available. Users migrating from `xtabond2` with system GMM should use difference GMM only.

2. **No multiple GMM variable groups**: `xtabond2` allows separate `gmm()` specifications with different lag structures. open-econs uses a single GMM-type block for all predetermined regressors.

3. **No `h()` HAC bandwidth**: The covariance matrix does not support a HAC bandwidth parameter for the weighting matrix.

4. **No `nested` iterated GMM**: The estimator does not iterate the weighting matrix to convergence.

5. **No forward orthogonal deviations**: Only first-difference transformation is implemented.

6. **No `artests` suppression**: The AR(1) and AR(2) tests are always computed and reported; they cannot be suppressed.

7. **No `sagan` option**: The Hansen J statistic is always reported; there is no separate Sargan statistic.

8. **No `predict()`**: The `ArellanoBondResult` does not implement a `predict()` method. Only `tidy()`, `summary()`, and `to_dict()` are available.

9. **No bootstrap**: Bootstrap standard errors are not implemented for AB.

## References

- Arellano, M., & Bond, S. (1991). Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations. *Review of Economic Studies*, 58(2), 277–297.
- Arellano, M., & Bover, O. (1995). Another Look at the Instrumental Variable Estimation of Error-Components Models. *Journal of Econometrics*, 68(1), 29–51.
- Blundell, R., & Bond, S. (1998). Initial Conditions and Moment Restrictions in Dynamic Panel Data Models. *Journal of Econometrics*, 87(1), 115–143.
- Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. *Econometrica*, 50(4), 1029–1054.
- Hausman, J. A. (1978). Specification Tests in Econometrics. *Econometrica*, 46(6), 1251–1271.
- Roodman, D. (2009). How to Do xtabond2: An Introduction to Difference and System GMM in Stata. *Stata Journal*, 9(1), 86–136.
- Stock, J. H., & Yogo, M. (2005). Testing for Weak Instruments in Linear IV Regression. In D. W. K. Andrews & J. H. Stock (Eds.), *Identification and Inference for Econometric Models: Essays in Honor of Thomas J. Rothenberg* (pp. 80–108). Cambridge University Press.
- Windmeijer, F. (2005). A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators. *Journal of Econometrics*, 126(1), 25–51.
