---
method: robust_reg
aliases:
  - robust regression
  - M-estimator
  - MM-estimator
  - bisquare regression
  - Tukey biweight
  - rreg
category: linear
api:
  - oe.robust_reg()
context_api: []
panel_api: []
problem: outlier- and heteroskedasticity-resistant linear estimation
estimator: Tukey biweight (bisquare) M-/MM-estimator of regression
stata_equivalent:
  - rreg
r_equivalent:
  - MASS::rlm
status: experimental
tier: 2
references:
  - huber1964
  - beatonkoenker1973
  - maronnayohai1979
  - yohai1987
  - rousseeuwleroy1987
---

# Robust Regression (M- and MM-Estimators) in Python

> **Estimator summary**: open-econs `robust_reg()` fits a linear model resistant
> to outliers and heteroskedasticity using redescending M-estimators of
> regression with the Tukey biweight (bisquare) ψ function (`c = 4.685`).
> **Stata `rreg` is the primary parity target** (default `parity="stata"`);
> R `MASS::rlm` is available as a toggle (`parity="rlm"`), exact to 1e-6.

## Overview

Ordinary least squares is highly sensitive to a small fraction of outliers: a
single bad observation can pull the entire fit. Robust regression replaces the
squared-error objective with a *redescending* loss whose influence function
down-weights (and ultimately rejects) points with large residuals.

**Key finding (verified 2026-07-19, the reason for this rework).** Stata
`rreg` is a bisquare **M**-estimator — *not* an MM-estimator. A previous agent
defaulted to R `MASS::rlm(method="MM")` and reported Stata `rreg` coefficients
diverging at ~1e-3, choosing to "follow R". Direct comparison of the committed
fixtures shows:

| Source | `_cons` | `x1` | `x2` |
|---|---|---|---|
| Stata `rreg` `e(b)` | 1.083157 | 2.459436 | -1.293816 |
| R `MASS::rlm(method="MM", bisquare)` | 1.084612 | 2.460500 | -1.294314 |
| R `MASS::rlm(method="M", bisquare, init="ls", scale="MAD")` | 1.083583 | 2.459780 | -1.293969 |

The MM default was the **wrong** parity target. The product must reproduce
Stata `rreg` by default. The residual ~4e-4 vs the plain-M R call is a
scale/init convention difference (Stata uses its own robust scale / Huber
init), not a bug to paper over.

Two estimators are exposed via the `method` toggle (rule 3 — optionality is a
feature):

* `method="mm"` (default) — bisquare estimator; for `parity="stata"` this is
  Stata's bisquare M-estimator, for `parity="rlm"` it is
  `MASS::rlm(method="MM")`.
* `method="huber"` — plain bisquare M-estimator (`MASS::rlm(method="M")`) with
  MAD scale, included for completeness.

The `parity` toggle (rule 15) selects the reference software:

* `parity="stata"` (DEFAULT) — pure-Python re-implementation of Stata
  `rreg.ado` v3.5.0 (OLS → Cook's-D drop → Huber init → bisquare IRLS →
  bias-correction regress). Coefficients and SEs match Stata `rreg` `e(b)`/`e(V)`
  to < 3e-10 (machine precision, within the 1e-6 rule). No R dependency.
* `parity="rlm"` — R `MASS::rlm` subprocess; coefficients + SEs + weights match
  R to 1e-6 (validated branch).

## Mathematical Formulation

### Population Model

\[
Y_i = X_i \beta + \varepsilon_i, \qquad i = 1, \dots, n
\]

where \(Y_i\) is scalar, \(X_i\) is a \(1 \times k\) row of regressors
(including a constant), and \(\varepsilon_i\) is an error term.

### M-Estimator of Regression

An M-estimator of regression solves the estimating equations

\[
\sum_{i=1}^n \psi\!\left(\frac{r_i}{s}\right) x_i = 0, \qquad r_i = y_i - x_i'\hat\beta,
\]

for a chosen influence function \(\psi\).  The scale \(s > 0\) is a robust
M-estimate of scale (below).  Equivalently this is an IRLS problem with working
weights

\[
w_i = \frac{\psi(r_i/s)}{r_i/s},
\]

so each IRLS step is weighted least squares with weights \(w_i\).

### Tukey Biweight (Bisquare) ψ and ρ

The bisquare (biweight) choice redescends to zero — points far enough out get
**zero** weight:

\[
\psi(u) = u \left(1 - \left(\frac{u}{c}\right)^2\right)^2 \mathbf{1}(|u| < c),
\qquad c = 4.685,
\]

with associated residual objective (ρ function)

\[
\rho(u) = \frac{c^2}{6}\left(1 - \left(1 - \left(\frac{u}{c}\right)^2\right)^3\right)
          \mathbf{1}(|u| < c) + \frac{c^2}{6}\,\mathbf{1}(|u| \ge c).
\]

The tuning constant \(c = 4.685\) gives the bisquare about 95% asymptotic
efficiency at the Gaussian model (the Stata `rreg` / R `MASS::rlm` default).

### Scale (Stata `rreg` MAD-type)

Stata `rreg` initialises from OLS, computes a Huber M-estimate initial step
(\(k = 1.345\)), then runs the bisquare M-estimator.  The scale \(s\) is a
robust MAD-type estimate re-computed each IRLS iteration:

\[
s^{(t)} = 1.4826 \cdot \operatorname{median}\!\big(|r^{(t)}_i -
\operatorname{median}(r^{(t)})|\big),
\]

where \(r^{(t)}\) are the current residuals.  The MAD scale is computed at the
**top** of each bisquare iteration from the previous iteration's residuals
(`scale = median(|r - median(r)|)/0.6745`), exactly as `rreg.ado` does; the
value carried into the bias-correction step is the last top-of-iteration scale
(= 0.9648544 for the rreg fixture).  This fully reproduces Stata's scale
iteration to machine precision.

### Key Quantities of Interest

* Coefficients \(\hat\beta\) (bisquare M/MM estimates)
* Robustness weights \(w_i \in [0,1]\) (final IRLS weights; outliers → 0)
* M-estimate scale \(s\)
* Standard errors, t-statistics, p-values, 95% confidence intervals (via the
  selected `vcov` branch)
* Residuals and fitted values; `rss`

## Assumptions

1. **Linearity**: \(E[Y \mid X] = X\beta\) (linear in parameters).
2. **Exogeneity**: \(E[\varepsilon \mid X] = 0\) (consistency of \(\hat\beta\)).
3. **No perfect collinearity**: \(\operatorname{rank}(X) = k\).
4. **Outliers are in the error**, not the design: bisquare M-estimators resist
   vertical outliers and heavy-tailed errors, but are *not* resistant to
   leverage-point (bad-\(X\)) outliers the way S-estimators with bounded ρ are.
5. **Inference**: the `vcov="stata"` covariance reproduces Stata's robust
   sandwich; `vcov="rlm"` reproduces R's `MASS::rlm` covariance.

## Estimator Derivation

The M-estimator is the fixpoint of the IRLS iteration

\[
\hat\beta^{(t+1)} = \arg\min_\beta \sum_i w_i^{(t)} (y_i - x_i'\beta)^2,
\qquad
w_i^{(t)} = \frac{\psi(r_i^{(t)}/s)}{r_i^{(t)}/s},
\quad r_i^{(t)} = y_i - x_i'\hat\beta^{(t)},
\]

with the scale \(s\) re-estimated (MAD-type) each step for `parity="stata"`.
Convergence is declared when \(\max_j |\hat\beta^{(t+1)}_j - \hat\beta^{(t)}_j|
< \texttt{acc}\) (default `1e-6`), capped at `maxit` (default 200).

## Inference

### Covariance Estimators (rule 15 toggle)

| Branch | Formula | Use case | Reference |
|--------|---------|----------|-----------|
| `vcov="stata"` (default for `parity="stata"`) | `V = (rss/(N-k)) (X_in' X_in)^{-1}`, rss = RSS of the Stata bias-correction regress | Reproduces Stata `rreg` `e(V)` to < 3e-10 | Stata `rreg` manual / `rreg.ado` v3.5.0 |
| `vcov="rlm"` (default for `parity="rlm"`) | `V = cov.unscaled * s^2` returned by `MASS::rlm` | Matches R `MASS::rlm` to 1e-6 | Venables & Ripley (2002) |

where \(W = \operatorname{diag}(w_i)\) and \(s\) is the M-estimate scale.

**Critical convention note (rule 15).** Stata `rreg` and R `MASS::rlm`
**disagree** on the point estimate: Stata uses a bisquare **M**-estimator with
its own robust scale; R's `method="MM"` adds an S-estimate refinement. Their
coefficients differ at ~1e-3. The product follows **Stata `rreg` by default**
(`parity="stata"`) and exposes R via `parity="rlm"` (both covered by tests):

* `parity="stata"` — pure-Python re-implementation of `rreg.ado` v3.5.0.
  Coefficients match Stata `e(b)` to **< 3e-10**; SEs match Stata `e(V)` to
  **< 3e-10**. The strict 1e-6 assertions PASS in
  `tests/stata/tests/test_stata_rreg.py` (gap ROBUST-REG-STATA resolved
  2026-07-19).
* `parity="rlm"` — R `MASS::rlm` subprocess, exact to **1e-6** on coefficients,
  SEs, and weights (validated branch).

### Two parity-critical subtleties (ROBUST-REG-STATA, resolved 2026-07-19)

These were the source of the ~1e-4 coefficient and ~2e-5 SE gaps and are now
encoded in `_stata_rreg_fit`:

1. **Carry-out weight, not re-evaluated weight.** The bias-correction step uses
   the LAST in-loop weight — the one actually used in the final reweighted
   `_regress [aw=weight]` — NOT a fresh weight re-evaluated at the updated
   residuals. Recomputing the weight from the final residuals breaks the WLS
   normal equations `X'(w*resid) = 0`, which makes the otherwise-exact bias
   correction (a genuine no-op at the WLS fixed point) drift the coefficients by
   ~1.2e-4.

2. **`lambda`'s `N` counts only non-zero-weight in-sample obs.** Stata's
   `regress [aw=weight]` DROPS observations whose analytic weight is exactly
   zero (the bisquare assigns `w = 0` to the 8 outliers with `|res/(tune*s)| >=
   1`). Therefore the `e(N)` carried into
   `lambda = 1 + ((df_m+1)/N)*(1-aa)/aa` is 192, not 200. Using the full
   in-sample `N` changes `lambda` by ~0.02% and pushes the correction RSS (hence
   `e(V)`) off Stata by ~2e-5. The correction regress `e(N)` itself is 200
   (all in-sample obs, including the zero-weight ones), so the VCE denominator is
   `N - k = 197`.

### Default Behavior

* `parity="stata"`, `method="mm"`, `vcov=None` (→ `vcov="stata"`) by default.
  This reproduces Stata `rreg` to the achievable tolerance.
* For R-exact 1e-6 parity, pass `parity="rlm"`.

### Technical Deviations from External Software

| Feature | open-econs (`parity="stata"`) | open-econs (`parity="rlm"`) | Stata `rreg` | R `MASS::rlm` |
|---------|-----------|-----------|--------------|---------------|
| Point estimate | bisquare M (rreg.ado v3.5.0) | bisquare MM | bisquare M | bisquare MM |
| Coef agreement | < 3e-10 vs Stata | exact vs R (1e-6) | ground truth | exact (vs OE rlm) |
| `vcov="stata"` SE | < 3e-10 vs Stata `e(V)` | — | `e(V)` | diverges |
| `vcov="rlm"` SE | — | exact vs R (1e-6) | diverges | `cov.unscaled*s^2` |
| Backend | pure-Python IRLS | R `MASS::rlm` subprocess | native | native |

## Implementation Details

### Formula Interface

Uses the [formulaic](https://github.com/matthewwardrop/formulaic) library
(R-style formulas). An intercept is included by default and is normalised to the
label `"(Intercept)"` (matching R / the fixtures).

```
y ~ x1 + x2            # bisquare M-estimator (Stata rreg parity)
y ~ x1 + x2 - 1        # no intercept
```

### Result Object

`RobustRegResult` (immutable via `BaseModel._freeze()`). Key attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.coefficients` | `pd.Series` | Bisquare M/MM estimates, named by term |
| `.std_errors` | `pd.Series` | Standard errors (selected `vcov` branch) |
| `.t_stats` | `pd.Series` | t-statistics |
| `.p_values` | `pd.Series` | Two-sided p-values (t with `df_resid`) |
| `.conf_int` | `pd.DataFrame` | 95% CIs (`lower`/`upper`) |
| `.weights` | `pd.Series` | Final robustness weights \(w_i\in[0,1]\) |
| `.scale` | `float` | M-estimate scale \(s\) |
| `.fitted_values` / `.residuals` | `pd.Series` | Fitted / residual values |
| `.method` / `.parity` / `.vcov` | `str` | Estimator / parity / covariance branch used |
| `.nobs` / `.df_resid` / `.df_model` | `int` | Dimensions |

Methods: `.tidy()`, `.summary()`, `.predict(newdata=None)`, `.vcov_matrix()`.

### Backend

`open_econs/core/_rlm_r.py` calls `MASS::rlm` once per fit via `Rscript`,
parsing a JSON payload (coefficients, covariance, scale, weights, residuals).
Used only by `parity="rlm"`. The `parity="stata"` path is fully pure-Python
(no R dependency).

## Stata / R Equivalents

### Stata

| open-econs | Stata | Notes |
|------------|-------|-------|
| `oe.robust_reg("y ~ x1 + x2", data=df)` | `rreg y x1 x2` | bisquare M, c=4.685, Huber init; coefs + SEs match to < 3e-10 |
| `oe.robust_reg("y ~ x1 + x2", data=df, vcov="stata")` | `rreg` `e(V)` | Bias-correction OLS VCE; SEs match to < 3e-10 |

### R

| open-econs | R | Notes |
|------------|---|-------|
| `oe.robust_reg("y ~ x1 + x2", data=df, method="mm", parity="rlm")` | `MASS::rlm(y~x1+x2, method="MM", psi=psi.bisquare, init="ls", scale.est="MAD")` | Exact to 1e-6 |
| `oe.robust_reg("y ~ x1 + x2", data=df, method="huber", parity="rlm")` | `MASS::rlm(y~x1+x2, method="M", psi=psi.bisquare, scale.est="MAD")` | Exact to 1e-6 |

## API Examples

### Default Stata `rreg` parity (primary product promise)

```python
import open_econs as oe

r = oe.robust_reg("y ~ x1 + x2", data=df)   # parity="stata" by default
print(r.tidy())
print(r.summary())
```

### R-exact covariance (validated 1e-6 branch)

```python
r = oe.robust_reg("y ~ x1 + x2", data=df, parity="rlm")
```

### Plain bisquare M-estimator

```python
r = oe.robust_reg("y ~ x1 + x2", data=df, method="huber", parity="rlm")
```

### Outlier diagnostics

```python
# Final robustness weights: outliers collapse toward 0.
low_weight = r.weights[r.weights < 0.1].index.tolist()
```

## Limitations

1. **Not leverage-point resistant**: bisquare M/MM resists vertical outliers
   but not high-leverage (bad-X) points as strongly as a full S-estimator with
   bounded ρ.
2. **`parity="stata"` now 1e-6**: the Stata `rreg` parity gap (ROBUST-REG-STATA)
   is **resolved** (2026-07-19). The pure-Python re-implementation of
   `rreg.ado` v3.5.0 reproduces `e(b)` and `e(V)` to < 3e-10. The two
   parity-critical subtleties (carry-out weight; `lambda`'s zero-weight `N`) are
   documented above and in `open_econs/models/linear/robust_reg.py`. Use
   `parity="rlm"` for validated R `MASS::rlm` 1e-6 parity.
3. **R dependency for `parity="rlm"`**: the exact R parity requires R + `MASS`.
   `parity="stata"` is pure-Python.
4. **No clustering / HAC**: only the bisquare robust sandwich is provided.

## References

- @huber1964
- @beatonkoenker1973
- @maronnayohai1979
- @yohai1987
- @rousseeuwleroy1987
