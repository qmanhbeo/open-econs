# open-econs vs statsmodels, linearmodels, and fixest — Feature Comparison

**open-econs is the only Python econometrics library that reproduces Stata and
R results to a verified numerical tolerance across the full empirical
economics and causal-inference stack.** This page is the long-form companion to
the README's short comparison: it shows, package by package, what each tool
actually covers.

Legend:
- ✅ = first-class in the named package.
- ⚠️ = partial / different scope, or only available through a separate
  ecosystem package rather than the named package itself.
- — = not offered.

> Note on the ⚠️ marks: `fixest` (R) ships `did2s` and `sunab`, so its
> Difference-in-Differences coverage is marked ✅. `fixest` has **no** synthetic
> control — R's `Synth` is a separate package — so that row is — for `fixest`.
> statsmodels offers some GMM and fixed-effects capability but not
> Arellano-Bond dynamic panels or proper two-way/re absorption, hence ⚠️.

## Side-by-side feature matrix

| Feature | statsmodels | linearmodels | fixest (R) | **open-econs** |
|---|:---:|:---:|:---:|:---:|
| OLS / WLS, robust & clustered SEs | ✅ | ✅ | ✅ | ✅ |
| Fixed effects (one/two-way, RE, FD) | ⚠️ | ✅ | ✅ | ✅ |
| IV / 2SLS | ✅ | ✅ | ✅ | ✅ |
| GMM & Arellano-Bond dynamic panels | ⚠️ | ✅ | — | ✅ |
| Logit / Probit / Multinomial logit | ✅ | — | ⚠️ | ✅ |
| Oaxaca-Blinder decomposition | — | — | ⚠️ | ✅ |
| Nonlinear least squares | ✅ | — | — | ✅ |
| Difference-in-Differences (staggered, Sun-Abraham, DID2S, event study) | — | — | ✅ | ✅ |
| Regression discontinuity (RDD) + density test | — | — | — | ✅ |
| Propensity-score & coarsened-exact matching + balance + Rosenbaum bounds | — | — | — | ✅ |
| Synthetic control + ADH permutation inference | — | — | — | ✅ |
| Time series: ARIMA / VAR / VECM / GARCH / unit-root & cointegration | ✅ | — | — | ✅ |
| **Numerical parity vs Stata / R (550+ tests, CI-enforced)** | — | — | — | ✅ |
| Reproducible exports (JSON / CSV / LaTeX / HTML) | ⚠️ | ⚠️ | — | ✅ |

## Why open-econs?

- **Stata/R parity, not just "close."** Coefficients and standard errors are
  checked against `xtabond2`, `csdid`, `teffects psmatch`, `rdrobust`,
  `synth_runner`, `dfuller` / `vars` / `arch`, and more — at ≤1e-6, and to
  machine precision for IV / Arellano-Bond / synthetic control. If a number
  differs from Stata or R, it is a bug, not a feature. This is enforced in CI:
  a numerical-equivalence regression fails the build before it ships.
- **One workflow for the whole empirical pipeline.** Data cleaning in pandas,
  estimation, diagnostics, sensitivity analysis, and publication-ready tables
  all live in Python — no more copy-pasting between Stata, R, and LaTeX.
- **Causal-inference-first.** DiD (staggered, Sun-Abraham, DID2S), RDD,
  matching with Rosenbaum bounds, and synthetic control with permutation
  inference are first-class, not afterthoughts.
- **Reproducible by construction.** Immutable results, named `pd.Series` /
  `pd.DataFrame` outputs, and one-call `.export()` to JSON, CSV, LaTeX, and HTML.

## See also

- [README](../README.md) — install, quick start, and the short comparison.
- [Release notes v1.0.3](release_notes_v1.0.3.md)
- [Migrating from Stata](migrating_from_stata.md) / [Migrating from R](migrating_from_r.md)
