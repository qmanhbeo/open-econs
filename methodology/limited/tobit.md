# Tobit (Censored Normal) Regression in Python — Stata `tobit` / R `AER::tobit` Parity

`open-econs` implements **Tobit (censored-normal) maximum-likelihood
regression** via `oe.tobit()`, a **hand-rolled MLE** over the Tobit
log-likelihood using `scipy.optimize.minimize`. `statsmodels` 0.14.6 has **no**
Tobit model, so — unlike `ologit`/`oprobit` — we cannot wrap a statsmodels
routine; the censored likelihood, its optimizer tolerances, and the OIM
covariance must be derived directly. `oe.tobit()` is reconciled to Stata base
`tobit` and R `AER::tobit` to a numeric tolerance of `1e-6`. This note records
the mathematics, the Stata-vs-R convention crosswalk (especially the
`sigma` / `Log(scale)` parameterization), the optimizer/tolerance decisions, and
the runnable commands, so future sessions do not re-derive them (standing rules
13, 16, 20).

---

## 1. The model

For an outcome `y_i` observed subject to censoring at limits `L` (left) and `R`
(right), the latent variable is

```
y_i* = x_i'β + u_i,    u_i ~ N(0, σ²)
y_i   = max(L, min(R, y_i*))
```

- **Left-censoring** (`y_i = L` whenever `y_i* <= L`): Stata default `tobit y x, ll(0)`;
  R `AER::tobit(y ~ x, left = 0, right = Inf)`.
- **Right-censoring** (`y_i = R` whenever `y_i* >= R`): Stata `tobit y x, ul(.)`;
  R `AER::tobit(y ~ x, right = .)`.
- **No censoring** (`L = -Inf, R = +Inf`): the Tobit MLE collapses to OLS on the
  (fully observed) outcome — used here as a smoke test.

The likelihood contribution of observation `i` is

```
L_i =  Φ((L - x'b)/σ)                     if y_i* <= L        (left-censored mass)
    =  (1 - Φ((R - x'b)/σ))               if y_i* >= R        (right-censored mass)
    =  φ((y_i - x'b)/σ) / σ               if L < y_i* < R      (uncensored density)
```

where `Φ`, `φ` are the standard-normal CDF / PDF. `β` and `σ` are concentrated
out by maximizing the summed log-likelihood.

---

## 2. Estimation — optimizer and tolerances (ROOT CAUSE, rule 16)

`scipy.optimize.minimize` over the `(β, ln σ)` parameterization (the log-scale
on `σ` keeps the optimizer in a well-conditioned, strictly-positive region):

1. **Start values**: OLS on the observed `y` gives `β₀`; `σ₀ = std(resid)`.
2. **Pass 1** — `BFGS` to a `gtol = 1e-8` neighborhood for a stable start.
3. **Pass 2 (polish)** — `L-BFGS-B` with `gtol = 1e-12`, `ftol = 1e-14`,
   `maxiter = 10000`. This is what lands the coefficients, `σ`, and OIM SEs at
   `1e-6` vs Stata/R. The raw `BFGS` default (`gtol ≈ 1e-5`) stops ~1e-5 short of
   Stata — **do not "simplify" the polish pass away** (rule 18 footgun).

### 2.1 σ / `Log(scale)` parameterization crosswalk (VERIFIED)

This is the single most error-prone convention difference (rule 15):

| Tool | What it prints for the scale | Stored quantity | Relation |
|------|------------------------------|-----------------|----------|
| **R `AER::tobit`** | `Log(scale)` and `Scale` | `summary()$sigma` = `σ` | `Log(scale) = ln σ` |
| **Stata `tobit`**   | `var(e.y)` in the coefficient table | 5th element of `e(b)` = `σ²` | `σ = sqrt(e(b)[5])`, `Log(scale) = 0.5·ln(e(b)[5])` |
| **OE `tobit()`**    | `r.sigma` and `r.log_scale` | both reported | `log_scale = ln(sigma)` |

**Source-verified numbers** (N=400, 147 left-censored at 0, `y_left ~ x1 x2 x3`):

- Stata `e(b)[5] = 1.042889` → `σ = sqrt(1.042889) = 1.021220`, `ln σ = 0.020998`.
- R `AER::tobit`: `Scale = 1.021`, `Log(scale) = 0.021`.
- OE: `sigma = 1.021219`, `log_scale = 0.020997`. **All agree to ~1e-6.**

> **Footgun (rule 18):** Stata's `tobit` does **NOT** store `e(sigma)` or
> `e(lnlnsigma)`. Its 5th `e(b)` element is the **variance** `σ²`, not `σ` and
> not `ln σ`. The fixtures extract `σ = sqrt(b[1,5])`. Any future Stata-probe
> that reads `e(sigma)` will silently get a missing value. Recorded here so it is
> never re-debugged.

---

## 3. Standard errors — OIM vs robust (OPEN GAP, rule 6/15/16)

- **OIM (nonrobust)** — `cov_type = "nonrobust"` (default) — is the **validated
  deliverable**. OE computes it as `inv(Hessian)` of the total negative
  log-likelihood, evaluated by `statsmodels.tools.numdiff.approx_hess` on the
  `(β, σ)` parameterization (NOT `(β, ln σ)` — the log-transform makes a flat,
  ill-conditioned direction at the optimum that breaks numeric Hessians, yielding
  `NaN`; see `methodology` root cause). This matches Stata/R OIM to `1e-6` on
  every coefficient and on `σ`.

  > **Why not the analytic expected-information or the sum-of-score-outer-products
  > (observed information)?** The per-observation observed information diverges
  > from Stata's OIM by ~2–3% at finite sample size (it is *not* what Stata
  > reports); the hand-derived analytic expected-information matrix for the
  > censored normal is error-prone (negative-definite blocks appeared in
  > derivation). The numeric **Hessian of the total NLL** (= Stata's `-Hessian` =
  > OIM) is exact and stable. Implemented as `inv(approx_hess(nll))`.

- **Robust / cluster (`HC0/HC1/HC2/HC3`, `cluster=`)** — a numerical-score
  sandwich. The per-observation score is differentiated numerically from the
  `(β, ln σ)` log-likelihood. This diverges from Stata's *exact* OIM-robust bread
  by **~1e-4** (same class of issue as `poisson`/`ologit`). The robust-SE branch
  is therefore **not asserted to 1e-6**; it is a documented open gap (see
  FUTURE_WORK.md). OIM parity is the shipped deliverable.

---

## 4. Predictions and margins

- `predict(type="ystar")` → `E[y* | x] = x'b` (latent linear predictor).
- `predict(type="y")` → `E[y | x]` (observed, censoring-aware):
  `x'b + σ·(λ_L − λ_R)` where `λ_L = φ((L−x'b)/σ)/Φ((L−x'b)/σ)` and `λ_R` is the
  analogous upper-tail inverse Mills ratio. Always lies in `[L, R]`.
- `predict(type="pr_gt0")` → `P(y > L | x) = 1 − Φ((L−x'b)/σ)`.
- `margins()` → average marginal effect of `x_k` on `E[y | x]`:
  `β_k · mean(P(y > L))` (the latent coefficient scaled by the probability of
  being uncensored) — the quantity Stata `margins` reports for a censored `tobit`.

---

## 5. Usage (Stata-manual style)

```python
import open_econs as oe

# Left-censored Tobit at 0 (matches Stata tobit y x1 x2 x3, ll(0))
r = oe.tobit("y ~ x1 + x2 + x3", data=df, left=0)
r.tidy()                 # coef / SE / z / p on the regressors (sigma in header)
r.summary()              # prints Log(scale) AND sigma (Stata crosswalk)
r.sigma                  # 1.021...
r.log_scale              # ln(sigma) = Stata "Log(scale)"
r.predict(type="y")      # E[y | x]  (observed, censoring-aware)
r.predict(type="ystar")  # E[y* | x] (latent)
r.predict(type="pr_gt0") # P(y > left | x)
r.margins()              # AME on E[y | x]

# Right-censored at 2, no left censoring
r2 = oe.tobit("y ~ x1 + x2 + x3", data=df, left=None, right=2.0)

# Robust / cluster SE (open gap vs Stata exact robust bread — see FUTURE_WORK)
r3 = oe.tobit("y ~ x1 + x2 + x3", data=df, left=0, cov_type="HC1")
r4 = oe.tobit("y ~ x1 + x2 + x3", data=df, left=0, cluster="group")
```

Reference equivalents:

```stata
tobit y x1 x2 x3, ll(0)
```

```r
library(AER)
tobit(y ~ x1 + x2 + x3, data = df, left = 0, right = Inf)
```

---

## 6. References

- Tobin, J. (1958). "Estimation of Relationships for Limited Dependent
  Variables." *Econometrica* 26(1).
- Amemiya, T. (1973). "Regression Analysis when the Dependent Variable is
  Truncated Normal." *Econometrica* 41(6).
- Greene, W. H. (2018). *Econometric Analysis* (ch. 19, Tobit / censored
  regression).
- Stata Corp. `tobit` — base Stata; reports `var(e.y) = σ²` as the 5th `e(b)`
  element and `Log(scale)` header as `ln σ`.
- R `AER::tobit` (Zeileis, Kleiber & Jackman 2008) — primary R reference
  (`censReg` is NOT installed on the dev box).
