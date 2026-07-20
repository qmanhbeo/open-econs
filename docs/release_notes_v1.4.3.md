# open-econs — PyPI Launch Notes

**Status: RELEASE 1.4.3.** `open-econs` **is already published on
PyPI** — the current released version is **1.4.2** (see
`docs/release_notes_v1.4.2.md`). This release builds on the already-published
1.4.2, adding the limited-DV (LDV) trust-hardening sprint and the new
`examples/` applied-micro demos. Per the project's Standard of Practice, the
tag/push/PyPI upload is performed by the release supervisor.

Target version: **1.4.3** (current `pyproject.toml` bumped from `1.4.2`).

## The parity thesis

open-econs is a Python library for empirical economics and causal inference that
reproduces **Stata and R results to a verified numerical tolerance**. The
overriding principle of the project is *parity with Stata and R is the product*:
every estimator option is audited against its reference implementation (Stata
`.ado`/Mata source or R package source), and a numerical mismatch fails the
build before it ships. The maximum tolerance for numeric parity is **1e-6**;
genuine reference divergences are exposed as explicit toggles, never papered
over.

## Breadth of coverage

- **40+ estimators** in one consistent, scikit-learn-style API: OLS/WLS, fixed
  effects (multi-way), IV/2SLS, GMM & Arellano-Bond, logit/probit/mlogit,
  Oaxaca-Blinder, nonlinear least squares, the full difference-in-differences
  family (Callaway-Sant'Anna, Sun-Abraham, Gardner DID2S, event studies),
  regression discontinuity, propensity-score & coarsened-exact matching,
  synthetic control with permutation inference, and a time-series module
  (ARIMA, VAR/VECM, GARCH, ARDL/UECM with the Pesaran-Shin-Smith bounds test,
  unit-root & cointegration).
- **550+ parity tests** (330+ vs Stata, 220+ vs R) run in CI on every release;
  a numerical mismatch fails the build. New methods are checked to ≤1e-6; IV,
  Arellano-Bond, and synthetic control reproduce reference results to machine
  precision.
- A full Stata/R → open-econs mapping and migration guides live under
  [`docs/`](docs/stata-r-mapping.md).

## This sprint: limited-DV trust-hardening

The limited-dependent-variable family's **robust and cluster standard errors**
were hardened to match Stata exactly (≤1e-6):

- **Poisson (PPML)** — non-clustered robust SE matches Stata `ppmlhdfe`'s
  Correia-Guimarães-Zylkin adjustment; cluster-robust matches to ≤1e-6.
- **Tobit (censored)** — OIM, robust, and cluster SEs match Stata `tobit`.
- **Ordered logit / probit** — OIM and robust (HC1, Stata `n/(n-1)`) SEs match Stata `ologit` / `oprobit`.
- **Negative binomial (`nbreg`)** — cluster SEs match Stata, with a Stata
  `constant`-dispersion convention toggle.

In addition, a **DFBETAS convention toggle** was added
(`Results.dfbetas(backend="stata_r")` vs `"statsmodels"`) so influence
diagnostics can be matched to either the Stata/R standardized convention or the
statsmodels convention.

See the [parity-coverage table in the README](../README.md#parity-coverage-validated-to-1e-6)
for the full validated matrix, and [`examples/`](../examples/) for runnable
applied-micro walkthroughs.

## Install

```bash
pip install open-econs                      # core estimators
pip install open-econs[plot]                # + matplotlib for .plot()
pip install open-econs[nls]                 # + sympy for nls()
pip install open-econs[dev,lint]            # + development & linting tools
```

> **Note:** `open-econs` **is published on PyPI** (latest = `1.4.2`). The
> commands above install the released version today. The limited-DV hardening and
> `examples/` added on `main` ship in **1.4.3** (this release). Until 1.4.3 is
> published, install the unreleased `main` from a local checkout with
> `pip install -e ".[dev]"`.

Requires Python ≥ 3.10.

## Where to look next

- [`README.md`](../README.md) — selling points, quick start, and the parity
  coverage table.
- [`examples/`](../examples/) — applied-micro examples (owned by a separate
  agent; not modified during release prep).
- [`docs/`](.) — methodology, migration guides, per-version release notes
  (`release_notes_vX.Y.Z.md`), and the Stata/R mapping.
