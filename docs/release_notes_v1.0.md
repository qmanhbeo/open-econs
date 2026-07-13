# open-econs v1.0.0 — Release Notes

**Status: PUBLISHED.** Tagged `v1.0.0`, GitHub Release and PyPI upload
completed. (v1.0.1 follows as a documentation-correction patch.)

## Highlights

- **Version `1.0.0`**, Development Status **5 — Production/Stable**.
- **API freeze** from `v1.0.0`: breaking changes to the public `__all__` API
  require a major version bump. See [`docs/api_stability.md`](docs/api_stability.md).
- **HAC (`cov_type="HAC"`)** is now available on all HAC-capable estimators:
  `ols()`/`reg()`, `fe()`, `nls()`, `iv()`, `gmm()`, `did()`, `event_study()`,
  and `PanelContext.driscoll_kraay()`. The canonical Newey-West estimators
  (`ols()`/`fe()`/`nls()`/`iv()`/`gmm()`/`did()`/`event_study()`) are validated
  against `statsmodels` / R `sandwich`.
- **Tutorials** for the core estimators (OLS, FE, IV, DiD) plus "Migrating from
  Stata" and "Migrating from R" guides.
- **Numerical parity CI** that runs Stata/R parity checks against committed
  fixtures on every release (zero skips on free runners).
- **Benchmark suite** (`benchmarks/ols_fe.py`) comparing `ols()`/`fe()` against
  `statsmodels` / `linearmodels`.

## ⚠️ `staggered_did()` HAC is experimental / non-canonical

`staggered_did()` now accepts `cov_type="HAC"` with a Newey-West temporal
correction applied to the aggregated influence function. **This is a project
convention, not an externally validated method.** Staggered-DiD HAC is a
contested area with no Stata/R reference implementation, so unlike the other
four HAC estimators (which rest on canonical literature), this one should be
treated as experimental.

- It raises a `UserWarning` when `cov_type="HAC"` is requested.
- At `lags=0` it reduces exactly to the entity-clustered standard error.
- See the estimator docstring for the full caveat. Do not present "HAC coverage
  complete" symmetrically across all estimators.

## Deferred items (explicitly not in v1.0)

- **Tutorials for RDD, PSM, synthetic control** — deferred as documented
  post-1.0 stubs in [`docs/tutorials/`](docs/tutorials/). The estimators
  themselves are shipped; only the walkthroughs are pending.
- **`nlogit()` (nested logit)** — recon complete (see `docs/nlogit-recon.md`)
  but not built: no validated fixture with τ∈(0,1), and the analytic gradient
  needs domain-expert implementation.
- **Stata/R fixture regeneration** — live regeneration of the committed parity
  fixtures from a real Stata/R install requires self-hosted runners. On free
  runners the parity *checks* run against the static committed fixtures (no
  binary launched, zero skips). This is a documented limitation, not a completed
  pipeline.

## NLS and the `sympy` dependency

`nls()` uses `sympy` for the analytic Jacobian (with a numerical fallback).
`sympy` is an **optional** extra (`pip install open-econs[nls]`); the NLS
parity tests that previously needed it now pass on CI because the `[dev]`
extra installs it. No hard dependency was added.

## Parity CI

`.github/workflows/ci-parity.yml` triggers on `release: published` and
`workflow_dispatch`, and runs the `stata`/`r`-marked suite against committed
fixtures. Two previously Stata-launched tests (`test_nls.py` `nl`,
`test_synth.py` `synth`) were converted to read committed R-derived fixtures,
so the whole parity suite runs with **zero skips** on free runners. The
`rddensity`-based RDD density tests need the optional `[rd]` python extra
(a pip package, not a binary), which the workflow installs.

## Benchmarks (real numbers, 10k-obs panel)

| estimator | open-econs (s) | reference (s) | speedup | max|Δcoef| |
|---|---|---|---|---|
| ols | 0.0136 | 0.0035 (statsmodels) | 0.25× | 0.00e+00 |
| fe  | 0.0171 | 0.0431 (linearmodels) | 2.51× | 1.28e-15 |

Honest reading: `ols()` matches `statsmodels` to machine precision but is ~4×
slower (statsmodels' OLS is C-optimized); `fe()` matches `linearmodels` to
machine precision and is ~2.5× faster.
