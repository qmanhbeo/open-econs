# Ordered Logit & Ordered Probit in Python — Stata `ologit` / `oprobit` and R `MASS::polr` Parity

`open-econs` implements **ordered logit** (`oe.ologit()`) and **ordered probit**
(`oe.oprobit()`) via `statsmodels.miscmodels.ordinal_model.OrderedModel`,
reconciled to Stata base `ologit` / `oprobit` and R `MASS::polr` (Venables &
Ripley 2002) to a numeric tolerance of `1e-6` on point estimates, cutpoints,
log-likelihood, and OIM standard errors. This note records the mathematics, the
Stata-vs-R-vs-statsmodels convention crosswalk (especially the cutpoint
parameterization), the MLE-polish root cause, and runnable commands so future
sessions do not re-derive them (standing rules 13, 16, 20).

---

## 1. The model (proportional odds / ordered probit)

For an ordered outcome `Y_i ∈ {0, 1, …, J}` and regressors `x_i`, the latent
variable is `y_i* = x_i'β + u_i` with `u_i ~ Logistic` (ologit) or `u_i ~ N(0,1)`
(oprobit). There are `J` cumulative thresholds (cutpoints) `c_1 < c_2 < … < c_J`
(the lowest and highest are `-∞`/`+∞`). The category probabilities are

```
P(Y_i = j | x_i) = F(c_j - x_i'β) - F(c_{j-1} - x_i'β)
```

with `c_0 = -∞` (`F = 0`), `c_J = +∞` (`F = 1`), and `F` the logistic or standard
normal CDF. This is the **Stata convention**: `P(Y ≤ j) = F(c_j - x'β)`. The
log-likelihood is the sum of the category probabilities over observations and is
maximized over `β` and the `J-1` free cutpoints.

---

## 2. Convention crosswalk (ROOT CAUSE, rule 16)

### 2.1 Cutpoint sign — NOT opposite (correcting a common assumption)

The task brief assumed Stata and `polr`/`OrderedModel` use *opposite* cutpoint
signs. **Source verification shows they do NOT.** All three store **cumulative,
increasing thresholds** `c_1 < c_2 < …` with `P(Y ≤ j) = F(c_j - x'β)`:

| Engine | Parameter | Example (ologit, this fixture) |
|--------|-----------|-------------------------------|
| Stata `ologit` | `e(b)` cut1/cut2/cut3 | -1.21539, 0.23989, 1.70875 |
| R `MASS::polr` | `zeta` | -1.21538, 0.23987, 1.70874 |
| statsmodels `OrderedModel` | `transform_threshold_params` | equal to Stata (after transform) |

statsmodels *internally* optimizes with a different parameterization
(`[c_1, log(c_2 - c_1), log(c_3 - c_2), …]` — only the first threshold is free,
the rest are exponentiated increments to enforce ordering). OE transforms those
back to the cumulative Stata convention via `model.transform_threshold_params`
and stores them in `.cutpoints` as `cut1, cut2, cut3`. **No sign flip is
applied or needed** — applying a negation would silently break Stata parity.
(Footgun, rule 18: do not "fix" the cutpoint sign; the common belief that polr
negates Stata is false for this parameterization.)

### 2.2 Verified numbers (probe: N=600, 4 categories, `y ~ x1+x2+x3`)

| Quantity | Stata `ologit` | R `polr` | OE (=statsmodels polished) | OE−Stata |
|----------|---------------|----------|----------------------------|----------|
| `b_x1` | 1.11103994 | 1.11102662 | 1.11103993 | 6e-9 |
| `cut1` | -1.21538554 | -1.21538029 | -1.21538546 | 8e-8 |
| `se_x1` (OIM) | 0.08854136 | 0.08854097 | 0.08854136 | 2e-9 |
| `ll` | -685.36118054 | -685.36118057 | -685.36118054 | 7e-13 |

OE matches **Stata to ~1e-7** (coefs/cutpoints/SEs) and **log-likelihood to
1e-13**. The Stata-vs-R gap on coefficients/cutpoints is ~1e-5 — an
**engine-level optimizer convergence difference**, not a formula difference
(both engines report the identical log-likelihood to 1e-8, and identical OIM SEs
to 1e-7). OE therefore anchors to Stata (the project's primary reference) and
records the Stata-R coef/cutpoint gap as a documented `skip` (rule 15), exactly
as the poisson iid gap is handled.

### 2.3 MLE-polish root cause (the 1e-6 achiever)

`statsmodels.OrderedModel.fit()` with its **default** optimizer (Nelder-Mead,
then BFGS) stops ~3e-5 short of Stata's coefficients because statsmodels' default
`gtol`/`ftol` are looser than Stata's Newton convergence. The log-likelihood
from the default fit already matches Stata to 1e-7, proving the MLE is the same
point — only the reported `params`/`bse` carry the 3e-5 imprecision.

**Fix (implemented, not loosened):** OE fits statsmodels once for structure,
then polishes the parameters with `scipy.optimize.minimize(..., method="L-BFGS-B",
options={"gtol": 1e-12, "ftol": 1e-14})` on the statsmodels negative
log-likelihood. The covariance is recomputed at the polished optimum: OIM from
`-inv(hessian)`, robust (HC0/HC1/HC2/HC3) from a numerical-score sandwich. After
polish, OE matches Stata to 1e-6 on all reported quantities. This is the
deliverable; the raw-statsmodels 3e-5 gap is never exposed to the user.

---

## 3. Reported quantities

- **Coefficients `β`** on the latent-index scale — match Stata to 1e-6.
- **Cutpoints `cut1…cut_{J-1}`** in Stata convention (cumulative, increasing).
- **OIM standard errors** (`cov_type="nonrobust"`) — match Stata/R to 1e-6.
- **Robust SEs** (`HC0`/`HC1`/`HC2`/`HC3`) — implemented as a sandwich with
  Stata's exact OIM bread `inv(-H)` and the **exact analytical observation
  scores** over the full `(β, cut1…)` vector in Stata's cumulative-cutpoint
  parameterization (Jacobian-transformed from statsmodels' incremental-exponential
  threshold params). HC1 matches Stata `vce(robust)` to ≤1e-6 (CLOSED, was open
  gap §4). Footgun (rule 18): OE's `HC1` uses Stata's `n/(n-1)` small-sample
  normalization to match `vce(robust)` — this is Stata's convention, not the
  generic `n/(n-k)` HC1 scaling. The OIM (nonrobust) SE — the original validated
  deliverable — is unchanged and still matches to 1e-6.
- **Log-likelihood** — matches Stata/R to 1e-6 (typically 1e-10+).
- **`.predict(type="probs")`** — per-category probabilities (sum to 1).
- **`.predict(type="class")`** — argmax category.
- **`.margins()`** — average marginal effects `d P(Y=j)/d x_k = -(β_k)·(f_{j-1} - f_j)`
  averaged over observations, where `f_j` is the latent-error PDF at
  `(cut_j - x'β)`. Margins sum to zero across categories (probabilities sum to 1).

---

## 4. Open gaps (rule 6/15/16 — documented, never loosened)

1. **Stata-vs-R coefficient/cutpoint divergence ~1e-5.** Inherent to the two
   reference engines' optimizers. OE matches Stata to 1e-6; R coef/cutpoint
   assertions are `skip`-ped in `tests/r/tests/test_r_ordered.py` with the exact
   magnitude noted. R log-likelihood and OIM SE *do* match OE to 1e-6 and are
   asserted.
2. **Robust (HC1) SE vs Stata `vce(robust)` ~4e-4 — RESOLVED (2026-07-20).**
   Root cause was the numerical-score bread: OE now uses Stata's exact OIM bread
   `inv(-H)` with the **exact analytical observation scores** (full `(β,
   cut1…)` vector in Stata's cumulative-cutpoint parameterization, via the
   Jacobian `d(nat)/d(sm)`), and Stata's `n/(n-1)` small-sample factor for HC1.
   `oe.ologit("y ~ x1+x2+x3", cov_type="HC1")` now matches Stata `ologit,
   vce(robust)` to ≤1e-6 (x1 5.8e-9, x2 2.8e-9, x3 1.3e-8). The `xfail(strict=True)`
   is removed and `tests/stata/tests/test_stata_ordered.py::TestStataOrderedRobustSE`
   is a real passing test. The OIM (nonrobust) SE — the original validated
   deliverable — is unchanged and still matches to 1e-6. Footgun (rule 18):
   OE's `HC1` small-sample factor is `n/(n-1)` (Stata's `vce(robust)`
   convention), not the generic `n/(n-k)` HC1 scaling.

---

## 5. Usage (Stata-manual style)

```python
import open_econs as oe

# Ordered logit, OIM SEs (matches Stata ologit / R polr logistic to 1e-6)
r = oe.ologit("y ~ x1 + x2 + x3", data=df)
r.tidy()              # coef / SE / z / p / CI table
r.cutpoints           # cut1, cut2, cut3 (Stata convention)
r.summary()
r.predict(type="probs")   # per-category probabilities
r.predict(type="class")   # predicted category
r.margins()           # AME on P(Y=j)

# Ordered probit
rp = oe.oprobit("y ~ x1 + x2 + x3", data=df)

# Robust SEs (HC1 = Stata vce(robust) convention; see open-gap note)
rr = oe.ologit("y ~ x1 + x2 + x3", data=df, cov_type="HC1")
```

Reference equivalents:

```stata
ologit y x1 x2 x3
oprobit y x1 x2 x3
```

```r
library(MASS)
polr(ordered(y) ~ x1 + x2 + x3, method = "logistic")   # ologit
polr(ordered(y) ~ x1 + x2 + x3, method = "probit")     # oprobit
```

---

## 6. References

- Stata manual: `ologit` / `oprobit` (base Stata, drops the first category as
  reference).
- Venables, W.N. & Ripley, B.D. (2002). *Modern Applied Statistics with S*
  (MASS `polr`).
- McCullagh, P. (1980). "Regression models for ordinal data." *JRSS B* 42(2).
- statsmodels 0.14.6 `miscmodels.ordinal_model.OrderedModel` source.
