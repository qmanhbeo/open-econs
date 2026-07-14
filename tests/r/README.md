# R Parity Tests — Maintainer Guide

## Dual-mode execution

These tests run in two modes:

| Mode | When | Behaviour |
|------|------|-----------|
| **Live R** | `OE_REGENERATE_FIXTURES` set **and** `R_EXE` points to a valid `Rscript` binary | Runs each `.R` file, regenerates the committed `.json` fixture |
| **Committed fallback** | No R, or `OE_REGENERATE_FIXTURES` unset (CI, contributors without R) | Reads the committed `.json` fixture directly — no R launched |

Both modes apply a **drift check**: if a `.R` file is newer than its `.json`
fixture, the test fails with `STALE FIXTURE`.  This catches the common mistake
of editing a `.R` but forgetting to regenerate the `.json`.

### Shared input convention

Unlike the Stata suite (which uses committed `.dta`), the R suite shares a
**single committed input CSV** between the Python test and the `.R` script:

- `tests/r/fixtures/<label>_input.csv` — the panel data, generated from the
  same deterministic builder the Python test uses.  **Both** the Python side
  (`_panel_from_csv`) and the `.R` script (`argv[1]`) read this exact file, so
  there is no cross-engine RNG-sync assumption.
- `tests/r/fixtures/<label>.json` — the expected output, written by the `.R`
  script (`argv[2]`) during regeneration.  The Python test reads it via
  `read_r(<label>)`.

So a full parity run compares the Python estimator fit (on the committed CSV)
against the R `Synth`/`nnet` fit (on the *same* committed CSV).  Any divergence
is a genuine cross-engine estimator difference, not a data-sourcing mismatch.

### Regenerating fixtures

When you change a `.R` file, regenerate its `.json` (and, if the data shape
changed, the input CSV) on a machine with R 4.6.1 + `Synth`/`jsonlite`/`nnet`:

```bash
# Via Python (launches Rscript if R_EXE is valid and the gate is set):
$env:OE_REGENERATE_FIXTURES=1
python -c "from tests.r.r_runner import run_r; run_r('synth_placebo_space')"

# Or directly:
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" `
   tests\r\do\synth_placebo_space.R `
   tests\r\fixtures\synth_placebo_space_input.csv `
   tests\r\fixtures\synth_placebo_space.json
```

Then commit both the `.R` (or input CSV) and the `.json`.

### CI behaviour

On GitHub Actions (ubuntu-latest, no R), `tests/r/` runs in committed-fallback
mode.  The `.json` fixtures are version-controlled, so the parity assertions
execute without R.  No R binary is launched and no fixture is rewritten.

**Two of the six R tests are re-gated** to `@pytest.mark.skipif(not R_AVAILABLE,
...)` and therefore **skip on CI** (which has no R):

- `test_synth_rank_deficient_qp_same_objective_different_w`
- `test_placebo_space_parity_r`

These were previously a narrow, deliberate exception gated behind `R_AVAILABLE`
because Python's `synth()` SLSQP fit was cross-OS nondeterministic for
rank-deficient panels. This was **fixed** by L2-regularizing the inner QP
(ridge 1e-12 when `N > P`), which makes the minimizer `W` unique and the outer
`V` landscape deterministic across BLAS backends
(`docs/synth-cross-os-solver-recon-update.md`).  The skipif guard was removed in
the same fix pass; both tests now run on CI against committed fixtures with zero
skips, like the other four R tests.

---

## Why the `synth` placebo `median_ratio` guard is `< 3.0`, not `< 1.0`

This is documented in `tests/test_synth_placebo.py::test_placebo_space_parity_r`
and is **not** a `time`-dtype bug.  Tracing `synth()` / `placebo_space()` shows
the numeric path never depends on the `time` column's dtype (an in-memory
`int64` cast of `time` reproduces the `object`-time result bit-for-bit in
W / V / the gap path).  The *only* difference between the in-memory builder
frame and the committed input CSV is that the CSV round-trips every float at
~1 ULP from the original (measured `max |delta| ~= 8.9e-16`).

For the rank-deficient placebo donors (documented nonconvex-V / solver-dependent
W: R's `kernlab::ipop` and our `scipy` SLSQP land on different local optima),
that 1-ULP input perturbation is amplified into a post/pre-MSPE ratio difference
of O(1–3) (measured `max |diff| ~= 4.6`, inside the `max_ratio < 5.0`
gross-regression guard; the permutation p-value still agrees exactly at `dp=0`).
The per-donor divergence is therefore expected to be O(1) in the typical case,
so the median guard tracks the documented "~3" divergence band rather than a
sub-unit threshold the in-memory builder only clears by luck of the 1-ULP
direction.  The p-value (primary correctness) and `max_ratio < 5.0` (gross
regression) guards are unchanged.

**Rule:** do not "fix" a failing `median_ratio` by coercing `time` to `object`
dtype in the test helper — that only reproduces the old number without a
mechanism.  If the median or max divergence regresses far beyond the documented
band, investigate the estimator, not the dtype.

---

## Tolerance guidelines

Cross-engine float precision between Python (`scipy`) and R (`Synth`/`nnet`) is
dominated by the nonconvex `V`-optimization path, not by BLAS noise:

| Estimator | Guard | Why |
|-----------|-------|-----|
| `synth` (W, pre-MSPE, gap path) | `rtol≈1e-6`–`1e-4` | Inner W problem is a unique convex QP — both engines agree closely |
| `synth` (`V`) | reported, not forced equal | Nonconvex outer loop; different local optima are expected (documented) |
| `synth` placebo-in-space ratios | `p_value` tight (`dp<0.05`, primary); `median_ratio < 3.0`; `max_ratio < 5.0` | Rank-deficient donors diverge O(1–3) by design; p-value is the robust reported statistic |
| `nls` (coef/se) | `rtol≈1e-4` | Same Gauss-Newton algorithm; float noise floor |
| `mlogit` (coef) | `rtol≈1e-3` | `nnet` one-vs-all vs OE multinomial; config-matched |

**Rule:** if a test needs a relaxed tolerance, document WHY in the test comment
(tie it to the documented nonconvex-V / solver-dependent behaviour).  Never
relax a tolerance as a shortcut to make a failing test pass.

## File naming convention

```
tests/r/generate-fixtures/
  {estimator}_{variant}.R          # e.g. synth_parity_default.R, nls_iid.R
tests/r/fixtures/
  {estimator}_{variant}_input.csv  # shared input panel (committed)
  {estimator}_{variant}.json       # expected output (committed)
```
