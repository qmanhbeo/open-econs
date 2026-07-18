# Poisson Fixed-Effects (PPML) Regression in Python — Stata `ppmlhdfe` / R `fixest::fepois` Parity

`open-econs` implements **Poisson pseudo-maximum-likelihood (PPML) regression
with high-dimensional fixed-effect absorption** via `oe.poisson()`, wrapping
`pyfixest.fepois` and reconciled to Stata's SSC `ppmlhdfe` (Correia, Guimarães &
Zylkin 2020) and R's `fixest::fepois` (Bergé 2018) to a numeric tolerance of
`1e-6`. This is the standard estimator for count outcomes and for
multiplicative-model / gravity-equation estimation (Santos Silva & Tenreyro
2006), where PPML is consistent even when the outcome is continuous and
non-negative. This note records the mathematics, the cross-tool convention
crosswalk (especially the cluster small-sample factor), and the runnable
commands, so future sessions do not re-derive them (standing rules 13, 16, 20).

---

## 1. The model

For a non-negative outcome `y_i` (`i = 1…N`), regressors `x_i`, and one or more
sets of fixed effects `α_{g(i)}` (`g = 1…G` FE dimensions), the conditional mean
is multiplicative:

```
E[y_i | x_i, α] = μ_i = exp( x_i'β + Σ_g α_{g(i)} )
```

`β` is estimated by **pseudo-maximum-likelihood** under the Poisson likelihood.
The score (first-order) conditions are the moment conditions

```
Σ_i ( y_i − μ_i ) x_i = 0                (for the regressors)
Σ_{i ∈ level ℓ of FE g} ( y_i − μ_i ) = 0   (for every FE level)
```

Crucially, **only the conditional mean has to be correctly specified** — the
Poisson variance assumption `Var = E` need not hold. The estimator is therefore
robust and is the workhorse for log-linear / gravity models (Santos Silva &
Tenreyro 2006, "The Log of Gravity").

### 1.1 Estimation algorithm (IRLS + alternating projections)

`ppmlhdfe`, `fixest::fepois`, and `pyfixest.fepois` all solve the same problem
with the **same algorithm** (this is *why* they agree to machine precision on
the point estimates):

1. **Iteratively Reweighted Least Squares (IRLS)** outer loop. At iteration `t`
   with current `μ_i^{(t)}`, form the working response
   `z_i = η_i + (y_i − μ_i)/μ_i` and weights `w_i = μ_i`.
2. Each IRLS step is a **weighted least squares** problem with the FE absorbed
   by **alternating projections** (Guimarães & Portugal 2010; Correia 2017) —
   the same demeaning core OE already uses for `oe.fe()` (pyfixest backend).
3. Iterate to convergence (`iwls_tol`, default `1e-8`).

Separation (regressors that perfectly predict `y_i = 0`) is detected and the
offending observations dropped; see §4.

---

## 2. Standard errors — the convention crosswalk (ROOT CAUSE, rule 16)

Point estimates from `pyfixest.fepois`, `fixest::fepois`, and `ppmlhdfe` agree
to `< 1e-7` out of the box. **Cluster-robust standard errors do NOT agree by
default** — this traces to the small-sample correction factor, and is a genuine
user-facing convention choice (rule 15).

### 2.1 Verified numbers (probe: N=500, 25 firms × 10 years, `cluster(firm)`)

| Tool / setting                                            | SE(x1)        |
|----------------------------------------------------------|---------------|
| `pyfixest.fepois` default  (= `fixest` default)          | 0.041639387   |
| `fixest::fepois` default                                 | 0.041639438   |
| **Stata `ppmlhdfe`**                                      | **0.0411779** |
| `pyfixest` `ssc(k_adj=False, G_adj=True, k_fixef="none")`| **0.041177878** |
| `fixest` `ssc(fixef.K="none", adj=FALSE, cluster.adj=TRUE)` | **0.041177929** |

The Stata/`fixest`-default ratio is a **constant scalar** `0.98892` on every
coefficient — i.e. a pure finite-sample factor, not a formula difference.

### 2.2 What each factor is

For CRV1 cluster-robust vcov `V = c · (X'X)⁻¹ (Σ_g X_g'e_g e_g'X_g) (X'X)⁻¹`,
the small-sample scale `c` is a product of up to three pieces:

- **`G_adj`** (cluster adjustment): `G/(G−1)` where `G` = number of clusters.
  *All three tools apply this.*
- **`k_adj`** (regressor dof): `(N−1)/(N−K)`. **`fixest`/`pyfixest` apply this
  by default; `ppmlhdfe` does NOT.**
- **`k_fixef`** (how absorbed-FE parameters count toward `K`): `fixest` default
  adds the number of fixed-effect coefficients into `K`; `ppmlhdfe` treats FE
  nested within the cluster as *redundant* (0 dof) and otherwise uses a
  different `K`. Setting `k_fixef="none"` removes fixest's FE contribution to
  `K`, matching `ppmlhdfe`'s handling for the nested case.

### 2.3 OE decision (rule 15 toggle)

`oe.poisson()` exposes **`vcov_backend`**:

- **`vcov_backend="fixest"` (DEFAULT)** — `pyfixest.fepois` defaults; matches R
  `fixest::fepois` to `1e-6`. Chosen as default because OE's compute backend
  *is* pyfixest and R parity is exact with no post-hoc rescaling.
- **`vcov_backend="stata"`** — passes
  `ssc(k_adj=False, G_adj=True, k_fixef="none")`; matches Stata `ppmlhdfe` to
  `1e-6`.

Both branches are covered by parity tests (rule 15). **Footgun (rule 18):**
`vcov_backend` only rescales the cluster/robust *variance*; it does NOT change
point estimates, deviance, or log-likelihood (those are identical across tools).
Do not "simplify" by hardcoding one factor — that silently breaks the other
engine's parity.

---

## 3. Reported quantities

- **Coefficients** `β` on the log (index) scale — directly comparable across all
  three tools.
- **IRR (incidence-rate ratios)** `exp(β)`, with delta-method SE
  `exp(β)·SE(β)`; matches Stata `ppmlhdfe, irr` and `poisson, irr`.
- **Deviance** `2·Σ[y log(y/μ) − (y−μ)]` — matches to `1e-6`.
- **Pseudo-loglik / pseudo-R²** — `ppmlhdfe` reports log-pseudolikelihood; equal
  to `fixest` `logLik` to `1e-6`.

---

## 4. Separation

Perfect predictors of `y=0` make the MLE non-existent for those coefficients.
`ppmlhdfe` implements Correia-Guimarães-Zylkin (2019) separation detection
(`fe`, `ir`, `simplex` methods) and drops separated observations. `pyfixest.fepois`
exposes `separation_check=["fe"]` / `["ir"]`. OE surfaces this via the
`separation_check` kwarg (default `None` = pyfixest default). **Flagged as an
open parity item** until three-way separation behaviour is tested; see
`FUTURE_WORK.md`.

---

## 5. Usage (Stata-manual style)

```python
import open_econs as oe

# Poisson FE, cluster-robust SE matching R fixest (default)
r = oe.poisson("y ~ x1 + x2", data=df, fixed_effects=["firm", "year"], cluster="firm")
r.tidy()          # coef table on log scale
r.summary()
r.irr()           # incidence-rate ratios exp(beta)
r.margins()       # average marginal effects on the count scale
r.predict()       # fitted conditional means mu_i

# Match Stata ppmlhdfe cluster SE exactly
r_stata = oe.poisson(
    "y ~ x1 + x2", data=df, fixed_effects=["firm", "year"],
    cluster="firm", vcov_backend="stata",
)
```

Reference equivalents:

```stata
ppmlhdfe y x1 x2, absorb(firm year) cluster(firm)
```

```r
library(fixest)
fepois(y ~ x1 + x2 | firm + year, data = df, cluster = ~firm)
```

---

## 6. References

- Santos Silva, J.M.C. & Tenreyro, S. (2006). "The Log of Gravity." *REStat*.
- Correia, S., Guimarães, P. & Zylkin, T. (2020). "Fast Poisson estimation with
  high-dimensional fixed effects." *Stata Journal* 20(1). (`ppmlhdfe`)
- Bergé, L. (2018). "Efficient estimation of maximum likelihood models with
  multiple fixed effects." (`fixest`)
- Guimarães, P. & Portugal, P. (2010). "A simple feasible procedure to fit
  models with high-dimensional fixed effects."
- Correia, S. (2017). "Linear models with high-dimensional fixed effects: an
  efficient and feasible estimator." (alternating-projections demeaner)
