# open-econs

[![PyPI version](https://img.shields.io/pypi/v/open-econs?color=blue)](https://pypi.org/project/open-econs/)
[![Python versions](https://img.shields.io/pypi/pyversions/open-econs)](https://pypi.org/project/open-econs/)
[![CI](https://github.com/qmanhbeo/open-econs/actions/workflows/ci.yml/badge.svg)](https://github.com/qmanhbeo/open-econs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/qmanhbeo/open-econs)](https://github.com/qmanhbeo/open-econs/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/open-econs)](https://pypi.org/project/open-econs/)

**open-econs is a Python library for empirical economics and causal inference
that reproduces Stata and R results to a verified numerical tolerance — built
for researchers migrating off Stata/R and for Python users who need causal
inference, it is the econometrics toolkit a Stata/R researcher would reach for,
but native in Python.**

Unlike statsmodels, linearmodels, or fixest, open-econs is validated against
Stata and R across the whole causal-inference stack: **550+ parity tests
(330+ vs Stata, 220+ vs R) run in CI on every release, and a numerical
mismatch fails the build before it ships.** New methods are checked to ≤1e-6;
IV, Arellano-Bond, and synthetic control reproduce reference results to machine
precision.

It covers 40+ estimators in one consistent API: OLS, fixed effects, IV/2SLS,
GMM & Arellano-Bond, logit/probit/mlogit, Oaxaca-Blinder, nonlinear least
squares, the full difference-in-differences family (Callaway-Sant'Anna,
Sun-Abraham, Gardner DID2S, event studies), regression discontinuity,
propensity-score & coarsened-exact matching, synthetic control with permutation
inference, and a time-series module (ARIMA, VAR/VECM, GARCH, ARDL/UECM with the
Pesaran-Shin-Smith bounds test, unit-root & cointegration).

## Install

```bash
pip install open-econs                      # core estimators
pip install open-econs[plot]                # + matplotlib for .plot()
pip install open-econs[nls]                 # + sympy for nls()
pip install open-econs[dev,lint]            # + development & linting tools
```
Requires Python ≥ 3.10.

## Quick start

```python
import open_econs as oe
import pandas as pd

df = pd.DataFrame({
    "income":    [30, 45, 55, 70, 85, 40, 60, 95],
    "education": [10, 12, 14, 16, 18, 11, 15, 20],
    "age":       [25, 30, 35, 40, 45, 28, 38, 50],
    "female":    [0,  0,  0,  0,  1,  1,  1,  1],
    "province":  ["A","A","B","B","C","C","A","B"],
})

r = oe.ols("income ~ education + age", data=df, cluster="province")
print(r.tidy())        # coefficient table (named pd.DataFrame)
print(r.summary())     # full OLS results

# Causal estimators use the same API:
oe.did_cs("y ~ x1 + x2 | group + time", data=panel_df)   # Callaway-Sant'Anna DiD
oe.psm("treat ~ x1 + x2", data=df)                        # propensity-score matching
oe.synth(...)                                            # synthetic control
```

Every result returns named `pd.Series` / `pd.DataFrame` outputs and exports to
JSON, CSV, LaTeX, or HTML with one call — `.export()`, `.to_latex()`,
`.to_html()`. Results are immutable after estimation.

## Applied-micro example (limited dependent variables)

open-econs reproduces the **robust and cluster standard errors** that applied
microeconometricians rely on daily — validated to ≤1e-6 against Stata:

```python
import open_econs as oe

# Poisson PPML — non-clustered robust SE matches Stata ppmlhdfe's
# Correia-Guimaraes-Zylkin adjustment; cluster-robust matches to <=1e-6.
r = oe.poisson("y ~ x1 + x2", data=df, vcov_backend="stata")
r = oe.poisson("y ~ x1 + x2", data=df, vcov_backend="stata", cluster="id")

# Tobit with robust SEs (matches Stata tobit, vce(robust))
r = oe.tobit("y ~ x1 + x2", data=df, cov_type="HC1")

# Ordered logit with heteroskedasticity-robust SEs (matches Stata ologit, vce(robust))
r = oe.ologit("y_ordered ~ x1 + x2", data=df, cov_type="HC1")
```

## Stata / R → open-econs (top mappings)

| Stata / R | open-econs |
|-----------|------------|
| `regress` (Stata) | `oe.ols()` |
| `xtreg, fe` (Stata) | `oe.fe()` |
| `ivregress 2sls` / `AER::ivreg` | `oe.iv()` |
| `xtabond2` (Stata) | `oe.abond()` |
| `csdid` (Stata) / `did` (R) | `oe.did_cs()` |
| `eventstudyinteract` (Stata) / `fixest::sunab` (R) | `oe.did_sa()` |
| `did2s` (Stata) / `fixest::did2s` (R) | `oe.did_gardner()` |
| `rdrobust` (Stata) | `oe.rdd()` |
| `teffects psmatch` (Stata) / `MatchIt` (R) | `oe.psm()` |
| `synth_runner` (Stata) / `synth` (R) | `oe.synth()` / `oe.placebo_space()` / `oe.placebo_time()` |

Full mapping: [docs/stata-r-mapping.md](docs/stata-r-mapping.md). Migration
guides: [Stata](docs/migrating_from_stata.md) / [R](docs/migrating_from_r.md).

## Parity coverage (validated to ≤1e-6)

Every estimator below is checked against its Stata and/or R reference in CI;
a numerical mismatch fails the build. The limited-DV family's **robust and
cluster standard errors** were hardened this release to match Stata exactly.

| Family | Stata anchor | R anchor | Robust / cluster SE |
|--------|--------------|----------|---------------------|
| OLS / WLS | `reg`, `newey` | `lm` | ✓ robust, cluster, HC0–HC3, HAC |
| Fixed effects (2-way) | `xtreg, fe` | `plm` | ✓ |
| IV / 2SLS | `ivregress 2sls` | `AER::ivreg` | ✓ robust, cluster, HC0–HC3 (HAC = R-reference) |
| Arellano-Bond (Diff + System GMM) | `xtabond2` | — | ✓ AR(1)/AR(2), Windmeijer 2-step |
| Poisson (PPML) | `ppmlhdfe` | `fixest::fepois` | ✓ cluster **and** non-clustered robust (CGZ) |
| Tobit (censored) | `tobit` | `AER::tobit` | ✓ OIM, robust, cluster |
| Ordered logit / probit | `ologit` / `oprobit` | `MASS::polr` | ✓ OIM **and** robust (HC1) |
| Negative binomial | `nbreg` | `glm.nb` / `fenegbin` | ✓ cluster; Stata `constant` dispersion toggle |
| Logit / probit / mlogit | `logit` / `probit` / `mlogit` | `glm` / `nnet` | ✓ |
| DiD (CS2021, Gardner, Sun-Abr, ES) | `csdid` / `did2s` / `eventstudyinteract` | `did` / `fixest` | ✓ |
| RDD | `rdrobust` | `rdrobust` | ✓ |
| PSM / CEM | `teffects psmatch` / `cem` | `MatchIt` / `cem` | ✓ |
| Synthetic control | `synth_runner` | `synth` | ✓ + permutation inference |
| ARDL/UECM, VAR, GARCH, ARIMA, unit-root | `ardl` / `var` / `arch` | `ARDL` / `vars` / `rugarch` | ✓ (TS CV vintage labelled) |
| Robust regression (M/MM) | `rreg` | `MASS::rlm` | ✓ bisquare M-estimator |

## Performance

v1.0.3 hardens hot loops with **bit-identical** vectorization/parallelization —
no parity tolerance loosened: `psm` is ~4× faster (k-NN + variance loops
batched), `_hac_S` Newey-West accumulation is a single `einsum`, and `did_cs`
bootstrap runs through an opt-in `parallel=` pool. GPU offload was evaluated
and deliberately declined (BLAS is already CPU-threaded). Full detail:
[docs/performance.md](docs/performance.md).

## Compare

open-econs is the only Python library covering the full causal-inference stack
*and* enforcing Stata/R parity. The side-by-side matrix against statsmodels,
linearmodels, and fixest is here:
[docs/comparison.md](docs/comparison.md).

## Documentation

- [Methodology](methodology/) — per-estimator math, assumptions, inference, and
  Stata/R equivalents.
- [Tutorials](docs/tutorials/README.md) — OLS, FE, IV, DiD, RDD, PSM, synthetic
  control walkthroughs.
- [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md) ·
  [v1.1.0 release notes](docs/release_notes_v1.1.0.md) ·
  [v1.0.3 release notes](docs/release_notes_v1.0.3.md)

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

## License

MIT
