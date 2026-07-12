# `synth()` — Cross-OS Solver Nondeterminism on Rank-Deficient / Nonconvex-V Panels

**Status:** Investigated, root-caused, **filed as a follow-up bug (NOT fixed in the fixture-migration work)**  
**Branch where exposed:** `feat/r-fixture-parity` (off `origin/main`)  
**Date filed:** 2026-07-12  
**PR:** #10  
**Linked roadmap entry:** `ROADMAP.md` line 293 (`synth()` shipped, with documented nonconvex-V solver divergence)

---

## Why This Was Investigated

As part of migrating the 6 `@pytest.mark.r` parity tests off live `Rscript`
calls onto committed JSON fixtures (so they run on CI without R), two tests
failed on CI (Linux) but passed locally (Windows):

1. `tests/test_synth.py::test_synth_rank_deficient_qp_same_objective_different_w`
2. `tests/test_synth_placebo.py::test_placebo_space_parity_r`

Both were initially assumed to be fixture-provenance problems (R fixture
generated on Windows, consumed on Linux CI). Root-cause investigation proved
they are **not** fixture problems — they are a genuine, latent bug in
`open_econs/models/causal/synth.py`'s cross-OS solver nondeterminism, exposed
only now because both tests were previously `@pytest.mark.skipif(not R_AVAILABLE)`
and therefore **skipped on CI** (CI has no R). The fixture migration removed that
skip, which is what surfaced the bug.

## Symptom (exact numbers gathered)

### Test #1 — rank-deficient QP covariate mismatch (`cov_mm_py`)

`cov_mm_py` is computed **purely from Python's own `_fit`** on the committed
input CSV; the R fixture is not involved in this assertion at all. It passes on
Windows and fails on CI:

| Environment | `cov_mm_py` | Result |
|-------------|-------------|--------|
| Windows (local, R present) | `2.67e-07` | PASS (`< 1e-2`) |
| CI (Linux, no R) | `3.27e-01` | FAIL (`< 1e-2`) |

`cov_mm_r` (the R fixture's own mismatch) is `3.81e-07` in **both** environments
— identical. So R is correct and OS-stable; Python's `scipy` SLSQP fit on this
rank-deficient donor pool (`P=2 < N=12`, inner QP is rank-deficient) lands on a
materially worse local optimum on Linux than on Windows.

### Test #2 — placebo-in-space ratio (`max|ratio|`)

A real cross-engine comparison. The permutation p-value matches exactly
(`dp=0`) on both OSes; only the per-donor post/pre-MSPE ratio diverges:

| Comparison | `max|ratio|` |
|------------|--------------|
| Windows-Python vs Windows-R fixture | `4.63` |
| Windows-Python vs **Linux**-R fixture (generated in WSL) | `4.6253` |
| CI (Linux-Python) vs Windows-R fixture | `7.97` |

**Decisive evidence that R is OS-stable:** Windows-Python gives `~4.6` against
*BOTH* the Windows-R and the Linux-R fixtures (difference `< 0.1%`). Therefore a
Linux-generated R fixture does **not** change the CI comparison — Linux-Python
vs Linux-R equals Linux-Python vs Windows-R, i.e. still `7.97`. The CI divergence
is driven entirely by Python's cross-OS fit, not by which OS produced the
fixture.

## Conclusion

Both failures share a **single root cause**: `synth()`'s nested V+W optimizer
(outer `V` via R `Synth`'s two-start equal/regression procedure, inner donor
weights `W` via `scipy.optimize.minimize` / SLSQP) is **cross-OS
nondeterministic** for rank-deficient / nonconvex-V panels. `scipy` SLSQP
follows a different optimization trajectory on Linux vs Windows (different BLAS
/ libomp / micro-arch), landing on a different local optimum whose objective
value can differ by O(0.1–1) in `cov_mm` and O(1–8) in placebo ratios. R's
`Synth` (kernlab `ipop`) is reproducible across OSes.

This is a **pre-existing latent defect** — it was always present; it was simply
hidden because the two affected tests were skipped on CI before the fixture
migration.

## Disposition (this task — PR #10)

Per the fixture-migration close-out decisions, the two tests are **re-gated**
back to `@pytest.mark.skipif(not R_AVAILABLE, ...)` (skip on CI, run where R is
present), with explanatory comments pointing here. They are **not** fixed by any
fixture change, and the tolerance is deliberately **not** relaxed to absorb the
OS divergence (that would mask the real bug). The other four R tests achieve the
migration's original zero-skip goal on CI.

**Do not** attempt to make these two pass on CI via:
- an OS-matched R fixture (proven ineffective — R is OS-stable),
- a third fixture variant, or
- an OS-specific / relaxed tolerance (masks the estimator bug; forbidden by
  `tests/r/README.md` tolerance rules).

## Fix Scope (future task — out of scope for PR #10)

Location: `open_econs/models/causal/synth.py` — the SLSQP / `_optimize_v` /
`_solve_w` path.

Candidate fixes (any one, verified cross-OS on Windows + Linux + CI):
1. Make the inner `W` QP solver deterministic across OSes — e.g. pin a
   deterministic SLSQP configuration (fixed `eps`, `maxiter`, seed-free but
   BLAS-independent path), or replace SLSQP with a deterministic QP solver
   (e.g. `quadprog` / `cvxpy` OSQP) for the inner weight problem.
2. For rank-deficient panels, pin a canonical minimizer (e.g. minimum-norm `W`
   over the flat objective subspace) so the result is unique regardless of
   solver trajectory.
3. Add a cross-OS CI job (Linux + a Windows/macOS runner, or a Linux matrix with
   a forced different BLAS) that asserts `cov_mm_py` and placebo `max|ratio|`
   agree across OSes to a tight tolerance, so this regression cannot recur
   silently.

## Affected tests

- `tests/test_synth.py::test_synth_rank_deficient_qp_same_objective_different_w`
- `tests/test_synth_placebo.py::test_placebo_space_parity_r`

Both currently: gated to R-present machines; expected to PASS on Windows-with-R,
FAIL on Linux-with-R, SKIP on CI (no R). After the estimator fix, both should
become fixture-based (R fixture) and run on CI again.
