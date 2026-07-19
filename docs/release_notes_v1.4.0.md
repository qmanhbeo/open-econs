# open-econs v1.4.0 — Release Notes

**Status: PUBLISHED.** Version bumped to `1.4.0` in `pyproject.toml` and
`open_econs/_version.py`; CHANGELOG, ROADMAP, FUTURE_WORK, and this document
updated. Tagged `v1.4.0`; the GitHub Release `published` event triggers the
trusted-publisher PyPI upload (`publish.yml`), and the release parity gate
(`ci-parity.yml`) runs on the same event.

## Highlights

- **Version `1.4.0`** — **quantile regression** and **outlier-robust (MM /
  bisquare) regression**, both with the project's Stata- and R-grade 1e-6
  parity discipline. Two new estimators join `open_econs.models.linear`, each
  shipping with Stata and R parity fixtures and methodology notes.
- New `open_econs/models/linear/quantile.py` (`quantile_reg`,
  `QuantileResult`) and `open_econs/models/linear/robust_reg.py`
  (`robust_reg`, `RobustRegResult`), exported from the top-level package.

## What's new

### Quantile regression — `oe.quantile_reg()`

- `method="qreg"` — point estimates via the Barrodale-Roberts simplex.
  Coefficients match Stata `qreg` and R `rq(method="br")` to 1e-6.
- `method="sqreg"` — simultaneous quantile regression (Stata `sqreg`).
- `method="bsqreg"` — bootstrap quantile regression (Stata `bsqreg`); the
  bootstrap is seed-controllable via `seed` (defaults to `None`, mirroring
  Stata's runtime behavior).
- `se_method="stata"` — analytical SEs matching Stata `e(V)` to 1e-6.
- `se_method="ker"` — kernel (Powell) SEs matching R
  `summary.rq(se="ker", hs=TRUE)` to 1e-6.
- Reference anchors: Stata `qreg` / `sqreg` / `bsqreg` + R `quantreg::rq`.

### Robust regression — `oe.robust_reg()`

- `parity="stata"` (default) — pure-Python bisquare M-estimator targeting
  Stata `rreg` (no R dependency). `method="mm"` / `method="huber"` select the
  M-estimator weighting.
- `parity="rlm"` — R `MASS::rlm` subprocess for exact 1e-6 parity to R.
- Reference anchors: Stata `rreg` + R `MASS::rlm`.

## Parity discipline

- Coefficients match the reference implementations to **1e-6** for both new
  estimators (`quantile_reg` vs Stata `qreg` AND R `rq(method="br")`;
  `robust_reg(parity="rlm")` vs R `MASS::rlm`).
- **Documented tolerance divergence (rule 15):** Stata `rreg` parity via the
  default `parity="stata"` lands coefficients at **~1.2e-4** and SEs at
  **~8e-4** relative to Stata, because Stata's internal scale iteration is not
  yet reverse-engineered. The strict 1e-6 coef/SE assertions are
  `xfail(strict=True)` (not loosened) and the gap is documented in
  `FUTURE_WORK.md` (`ROBUST-REG-STATA`). Use `parity="rlm"` for exact R
  parity when Stata `rreg` fidelity is not required.
- `quantile_reg` bootstrap (`bsqreg`) is seed-controllable; the default
  `seed=None` is intentionally non-reproducible to mirror Stata.
- Full suite green before release (excluding `synth_placebo`), ruff + mypy
  clean on `open_econs/`.

## Scope

v1.4.0 is a **feature release**. No breaking changes. The documented Stata
`rreg` coef/SE residual gap is the only open parity item introduced; it is
flagged, not hidden.

## Upgrade

```bash
pip install -U open-econs
```

No breaking changes. Deprecation shims from v1.0.2 remain in place and will be
removed in v2.0.0.
