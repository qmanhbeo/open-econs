# `synth()` — Cross-OS Solver Nondeterminism: Fix Update

**Status:** Fixed.  
**Branch:** `fix/synth-cross-os-solver`  
**PR:** #19  
**Previous doc:** `synth-cross-os-solver-recon.md` (root cause investigation)

---

## The Fix

`docs/synth-cross-os-solver-recon.md` correctly established the root cause:
`synth()`'s inner QP (`_solve_w`) uses a singular Hessian when `N > P`
(donor count > predictor count).  The minimizer `W` is then not unique —
there is a `(N-P)`-dimensional affine subspace of optimal `W` vectors.  Because
`scipy`'s SLSQP (and its underlying BLAS) selects a `W` from this subspace in a
floating-point-dependent way, the outer `V` optimizer (`_fn_v` → `_optimize_v`)
receives a slightly different `fn_v` value at the same `V` on different OS
BLAS backends.  This difference cascades through SLSQP's finite-difference
gradient computation, landing on materially different local `V` optima
(Windows: `cov_mm ≈ 2.67e-07`; CI Linux: `cov_mm ≈ 0.327`).

### What Changed

One line in `_solve_w()` (`synth.py:434-435`):

```python
if N > P:
    H = H + 1e-12 * np.eye(N)
```

Adding a tiny L2 ridge (1e-12) to the inner QP Hessian when `N > P` makes
the problem **strictly convex** — the minimizer `W` is unique for any given
`V`.  The ridge is so small that:
- **Well-determined cases** (`P >= N`): no ridge added at all; results are
  **bit-identical** to the pre-fix code.
- **Rank-deficient cases** (`N > P`): the unique `W` is the minimum-norm
  element of the former optimal affine subspace.  This changes which `V`
  the outer optimizer converges to (from `[0.014, 0.986]` to
  `[0.909, 0.091]` for the underdetermined fixture), but both achieve
  `cov_mm < 1e-2` (post-fix: `5.59e-05`) and the outcome is now
  **deterministic across BLAS backends** — the same on any OS.

### Why Not a Larger Change

Several approaches were evaluated and rejected:

1. **Minimum-norm `W` via second QP solve** — principled but expensive;
   the L2 ridge approximates this at negligible cost.

2. **Replace SLSQP with a derivative-free outer optimizer (Nelder-Mead,
   COBYLA, or Powell)** — problem is that the global minimum of the outer
   objective `fn_v` (pre-treatment MSPE) is at boundary `V` values
   (e.g. `[0.0, 1.0]` for the underdetermined fixture) where `cov_mm`
   can be large (`0.327`).  SLSQP's gradient-based trajectory happens to
   converge to a local minimum with better covariate balance, which we
   preserve — we just make that trajectory OS-deterministic.

3. **Add `cvxpy`/`osqp` dependency** — unnecessary; the L2 ridge requires
   no new dependencies and is a one-line change.

4. **Use `ipop` (kernlab) like R** — not portable / would add an R dependency.

### Validation

- **Windows (local)**: All 28 synth and placebo tests pass.
- **Windows stability**: 10 consecutive runs of the rank-deficient case are
  bit-identical (`cov_mm` standard deviation = 0.0).
- **Well-determined regression**: `_make_panel_welldetermined` (P=12, N=12)
  unchanged: max `|W_diff| = 0.0` vs pre-fix.
- **CI (Linux)**: PR CI confirms all tests pass on ubuntu-latest with
  Python 3.12 and 3.13, including the previously-skipped
  `test_synth_rank_deficient_qp_same_objective_different_w` and
  `test_placebo_space_parity_r`.

### Files Changed

| File | Change |
|------|--------|
| `open_econs/models/causal/synth.py` | Added L2 ridge in `_solve_w()` when `N > P` |
| `tests/test_synth.py` | Removed `skipif(not R_AVAILABLE)` from rank-deficient QP test |
| `tests/test_synth_placebo.py` | Removed `skipif(not R_AVAILABLE)` from placebo-space parity test |
| `tests/r/README.md` | Updated exception section to reflect fix |
| `ROADMAP.md` | Marked synth cross-OS item as done |
| `docs/synth-cross-os-solver-recon-update.md` | This file |

### Still Open (unchanged)

- `nlogit()` — recon complete, documented in `docs/nlogit-recon.md`;
  implementation blocked on `mlogit`-spec equality-structure syntax.
- API freeze candidate — no more breaking signature changes without
  deprecation.
- Full docstring coverage + type-checked public API.
- GEE — candidate for v1.0+ if demand materialises.
