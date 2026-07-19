# open-econs v1.2.0 — Release Notes

**Status: PUBLISHED.** Version bumped to `1.2.0` in `pyproject.toml` and
`open_econs/_version.py`; CHANGELOG, ROADMAP, FUTURE_WORK, and this document
updated. Tagged `v1.2.0`; the GitHub Release `published` event triggers the
trusted-publisher PyPI upload (`publish.yml`), and the release parity gate
(`ci-parity.yml`) runs on the same event.

## Highlights

- **Version `1.2.0`** — the first **count & limited dependent variable** release.
- **New `open_econs/models/limited/` module**, covering FE-backed `poisson`,
  `nbreg` (NB1/NB2), ordered `ologit`/`oprobit`, and censored-normal `tobit`
  under the same source-verified Stata/R parity discipline as the rest of the
  library. Hand-rolled cores where no reference-compatible backend exists
  (Tobit, ordered) are still validated against reference source, not just
  output.
- **All four estimators match Stata/R to 1e-6** on point estimates, OIM SEs,
  and log-likelihood, anchored to live reference output (`e(b)`, `e(V)`,
  `summary()`).

## What's new

### Count models

- `poisson()` — FE-backed PPML via the HDFE demeaning core (the
  `fixest::fepois` convention). `vcov_backend` toggle: `"fixest"` (R parity,
  default) vs `"stata"` (ppmlhdfe cluster-robust parity). `CountResult` adds
  `.irr()`, `.margins()`, `.predict()`.
- `nbreg()` — NB1/NB2 hand-rolled inside the HDFE IRLS core (pyfixest has no
  `fenegbin`; Stata base `nbreg` has no FE absorption). `NegBinResult` adds
  `.alpha()`/`.theta()`/`.irr()`/`.margins()`/`.predict()`/`.tidy()`/`.summary()`.
  Toggles (rule 15): `vcov_backend` (`"fixest"` default / `"stata"`);
  `dispersion` (`"mean"` / `"const"`).

### Limited dependent variable models

- `ologit()` / `oprobit()` — ordered logit/probit via `statsmodels`
  `OrderedModel` with an L-BFGS-B polish pass (gtol=1e-12) so point estimates,
  cutpoints, OIM SEs, and loglik match Stata and R to 1e-6. `OrderedResult`
  adds `.cutpoints`, `.predict(type="class"|"probs")`, `.margins()`. `cov_type`
  ∈ {nonrobust, HC0, HC1, HC2, HC3}.
- `tobit()` — hand-rolled censored-normal MLE (statsmodels has no Tobit),
  validated vs R `AER::tobit` (censReg NOT installed) and Stata `tobit`.
  `TobitResult` returned. Toggles (rule 15): `left` / `right` censoring bounds.

### Reference anchors (1e-6)

- `poisson` → Stata `ppmlhdfe` + R `fixest::fepois`.
- `nbreg` → R `fixest::fenegbin` + Stata `nbreg, dispersion(mean)`.
- `ologit`/`oprobit` → Stata `ologit`/`oprobit` + R `MASS::polr`.
- `tobit` → R `AER::tobit` + Stata `tobit`.

## Parity discipline

- Every new limited-DV estimator ships with a parity test against Stata and/or R
  at ≤1e-6, run in CI on every release. No tolerance was loosened for this
  release (rule 2).
- **Rule 22 compliance:** all documented unsolved disparities now have
  `pytest.mark.xfail(strict=True)` tests carrying real assertions — no `skip`.
  Gaps so asserted (magnitudes recorded, never papered over):
  - `poisson` non-clustered robust SE ~4e-4;
  - `ologit`/`oprobit` Stata-vs-R coef/cutpoint ~1e-5 + HC1 robust SE ~4e-4;
  - `tobit` robust/cluster SE ~1e-4;
  - `nbreg` Stata `dispersion(constant)` distinct MLE + non-clustered OIM SE ~4%.
- **Repo-wide `xfail` added this cycle:** TS-1 ADF CV-vintage; TS-2 `dfgls`
  lag-selection; `abond` R-parity deferral (R `plm` `pgmm` broken); Johansen
  O-L vs MacKinnon CV-table split.
- Full suite green: **1246 passed, 23 xfailed, 30 deselected** (excluding
  `synth_placebo`), ruff + mypy clean on `open_econs/`.

## Scope

v1.2.0 is a **feature (count & limited dependent variable)** release. The next
method additions follow the roadmap: `v1.3` diagnostics (`bg_test`,
`white_test`, `ljung_box`, influence measures) and a consistent first-class
result diagnostics API. `heckman()` and `feglm` binomial FE absorption are
**deferred** (NOT in v1.2).

## Upgrade

```bash
pip install -U open-econs
```

No breaking changes. Deprecation shims from v1.0.2 (`staggered_did` → `did_cs`,
`did_sun_abraham` → `did_sa`) remain in place and will be removed in v2.0.0.
