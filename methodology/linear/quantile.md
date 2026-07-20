---
method: quantile
aliases:
  - quantile_reg
  - qreg
  - median_regression
category: linear
api:
  - oe.quantile_reg()
context_api: []
panel_api: []
problem: Quantile (including median) regression by Koenker & Bassett (1978): estimate the conditional τ-th quantile of y given x via the check-function loss, with a convention-correct VCE.
estimator: Quantile regression. Coefficients from the exact linear-programming solution of the check-function objective (same vertex as the Barrodale-Roberts simplex used by Stata qreg and R quantreg::rq(method="br")). Standard errors follow a rule-15 toggle between Stata's i.i.d. sparsity sandwich and R's Powell kernel sandwich, or a paired bootstrap for the sqreg/bsqreg methods.
stata_equivalent:
  - qreg (Stata 17; default vce(iid, fitted hsheather))
  - bsqreg (single-quantile bootstrap, reps=20)
  - sqreg (simultaneous-quantile bootstrap)
r_equivalent:
  - quantreg::rq (R package quantreg, v6.1; method="br")
  - quantreg::summary.rq (se="ker", hs=TRUE)
status: mature
tier: 1
references:
  - Koenker & Bassett (1978), "Regression Quantiles", Econometrica 46(1), 33-50.
  - Barrodale & Roberts (1974), "Solution of an overdetermined system of equations in the ℓ₁ norm", CACM 17(6), 319-320.
  - Koenker (2005), Quantile Regression, Cambridge University Press.
  - Hall & Sheather (1988), "Estimating the quantiles of a smooth distribution function", JoASA 83(404), 1203-1213.
  - Powell (1991), "Estimation of monotonic regression models under quantile restrictions", in Nonparametric and Semiparametric Methods in Econometrics.
---

# Quantile Regression — `oe.quantile_reg()` — Stata `qreg` / R `quantreg::rq` Parity

## API

```python
oe.quantile_reg(formula, data,
                tau=0.5, method="qreg",
                reps=20, seed=None,
                se_method="stata", cov_type="nonrobust")
```

| argument    | meaning |
|-------------|---------|
| `tau`       | quantile `τ ∈ (0,1)`; default `0.5` is the median (matches Stata `qreg` default). |
| `method`    | `"qreg"` analytic sparsity/kernel sandwich; `"bsqreg"` single-quantile paired bootstrap; `"sqreg"` simultaneous-quantile bootstrap (single-`τ` here == `bsqreg`). |
| `se_method` | **rule-15 toggle** for `method="qreg"`: `"stata"` (default) or `"ker"` (R kernel). |
| `reps`      | bootstrap replications (Stata default 20). |
| `seed`      | RNG seed; **required** for reproducible bootstrap SEs. |

## Coefficient computation

Quantile regression minimises the check-function loss

```
min_b  Σ_i  ρ_τ(y_i − x_i'b),   ρ_τ(u) = u·(τ − 1{u<0}).
```

`statsmodels.QuantReg` uses an interior-point/IRLS solver that lands ~1e-5 off
the simplex vertex except at the unique median, so we do **not** wrap it. Instead
we solve the equivalent LP exactly via SciPy `linprog(method="highs")`:

```
min  Σ_j [0·(b⁺,b⁻) + τ·u⁺ + (1−τ)·u⁻]
s.t. X b⁺ − X b⁻ + u⁺ − u⁻ = y,  b⁺,b⁻,u⁺,u⁻ ≥ 0,
     b = b⁺ − b⁻.
```

HiGHS returns the same *basic* (vertex) solution the Barrodale-Roberts simplex
finds when the optimum is unique, so coefficients reproduce Stata `qreg` and R
`rq(method="br")` to machine precision (verified at `τ ∈ {0.25, 0.5, 0.75}`).

## Standard-error conventions (rule-15 toggle)

Two valid conventions disagree; both are exposed and both tested.

### `se_method="stata"` (default) — Stata `qreg` default VCE

Stata `qreg` (without `vce(robust)`) uses the i.i.d. sparsity sandwich

```
V = s² · τ(1−τ) · (X'X)⁻¹
```

with the **fitted** sparsity estimate

```
s = mean_k ( x_k' (b̂(τ+h) − b̂(τ−h)) ) / (2h)
```

and the Hall-Sheather bandwidth

```
h = n^{−1/3} · z_{α/2}^{2/3} · ( 1.5·f₀² / (2·x₀² + 1) )^{1/3},
   x₀ = Φ⁻¹(τ),  f₀ = φ(x₀),  z_{α/2} = Φ⁻¹(1−α/2).
```

`b̂(τ±h)` are the QR estimates at the perturbed quantiles (h shrunk if `τ±h`
leaves `(0,1)`). This reproduces Stata `qreg`'s `e(V)` diagonal to **≤ 1e-6**
(verified `τ=0.5` to 1e-8, `τ=0.25` to 2e-8).

> Footgun: Stata offers an alternative `vce(robust)` (Powell kernel) that
> matches R's `se="ker"`, not the `vce(iid, fitted)` default. `se_method="stata"`
> tracks the *default* `qreg` output the analyst sees without options.

### `se_method="ker"` — R `quantreg::summary.rq(se="ker", hs=TRUE)`

R's Powell (1991) kernel sandwich with the Hall-Sheather bandwidth rescaled by
the residual spread:

```
h  = ( Φ⁻¹(τ+h₀) − Φ⁻¹(τ−h₀) ) · min( sd(û), IQR(û)/1.34 ),
f_i = φ(û_i / h) / h,
V  = τ(1−τ) · (X'F X)⁻¹ · X'X · (X'F X)⁻¹,   F = diag(f_i).
```

This reproduces R's `summary.rq(se="ker", hs=TRUE)` to **≤ 1e-6** (verified at
`τ ∈ {0.25, 0.5}`).

> Footgun: `se_method` only affects `method="qreg"`. The bootstrap methods
> (`sqreg`/`bsqreg`) ignore it.

## Bootstrap methods (`sqreg`, `bsqreg`)

Paired `(y_i, x_i)` resampling with replacement, `reps` draws, QR refit each
draw; the covariance of the bootstrap coefficient draws is the VCE. Coefficients
are deterministic (BR simplex), so they match Stata exactly. The **bootstrap
SEs** are compared to a **documented tolerance only** (1e-2 in the parity test):
Stata's bootstrap RNG is not portable, so the Python side uses NumPy's
`default_rng(seed)` instead of replicating Stata's internal sampler.

`sqreg` with a single `τ` is numerically identical to `bsqreg`; full
between-quantile cross-blocks of multi-quantile `sqreg` are out of scope.

## Stata / R mapping

| open-econs                | Stata                          | R quantreg                          |
|---------------------------|--------------------------------|-------------------------------------|
| `quantile_reg(..., tau)`  | `qreg y x, quantile(τ)`        | `rq(y ~ x, tau=τ, method="br")`     |
| `se_method="stata"`       | `qreg` default `e(V)` (iid)    | — (Stata-only convention)           |
| `se_method="ker"`         | `qreg, vce(robust)`            | `summary.rq(fit, se="ker", hs=TRUE)`|
| `method="bsqreg", seed`   | `bsqreg y x, q(τ) reps(r)`     | `boot.rq(...)` (not asserted exact) |
| `method="sqreg", seed`    | `sqreg y x, q(τ) reps(r)`      | `boot.rq(...)` (not asserted exact) |

## Parity verification

- **Coefficients**: identical to both Stata `qreg` and R `rq(method="br")` to
  machine precision (`tests/stata/tests/test_stata_qreg.py`,
  `tests/r/tests/test_r_qreg.py`).
- **SE `se_method="stata"`**: matches Stata `qreg` `e(V)` diagonal to ≤ 1e-6.
- **SE `se_method="ker"`**: matches R `summary.rq(se="ker", hs=TRUE)` to ≤ 1e-6.
- **Bootstrap**: coefficients exact; SEs within the documented 1e-2 tolerance.

## Divergences / flags

- `statsmodels.QuantReg` is **not** a valid reference (IRLS solver; ~1e-5 off
  the simplex vertex away from the median). Wrapped only in a non-parity unit
  test at `τ=0.5` with a widened tolerance.
- Bootstrap SEs are not bit-reproducible across Stata/R/Python (RNG
  implementation), hence the tolerance. Coefficients remain exact.
