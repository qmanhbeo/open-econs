# `synth()` — Cross-OS Solver Nondeterminism: Status Update

**Status:** Partially mitigated — Component 1 (rank-deficient inner QP) is
**xfail-gated, not algorithmically fixed**; Component 2 (placebo parity) is
**validated green**. CI restructured so push/PR runs only the fast suite (~10
min observed) with the slow synth sweeps moved to a nightly + manual safety net.
**Branch:** `fix/synth-cross-os-solver`
**PR:** #19 (OPEN — do not merge without explicit instruction)
**Previous docs:**
- `synth-cross-os-solver-recon.md` — root-cause investigation
- `synth-cross-os-solver-recon-update.md` — superseded by this file

---

## Component 1 — rank-deficient inner QP: xfail-gated (deliberate, final)

`docs/synth-cross-os-solver-recon.md` established the root cause: `synth()`'s
inner QP (`_solve_w`) uses a singular Hessian when `N > P` (donor count >
predictor count). The minimizer `W` is then not unique — a `(N-P)`-dimensional
affine subspace of optimal `W` vectors. `scipy`'s SLSQP (and its underlying
BLAS) selects a `W` from this subspace in a floating-point-dependent way, so the
outer `V` optimizer receives a slightly different `fn_v` value at the same `V`
on different OS BLAS backends, cascading to materially different local `V`
optima (Windows: `cov_mm ≈ 2.67e-07`; CI Linux: `cov_mm ≈ 0.327`).

### What was changed (and its measured effect)

1. `synth.py:434-435` — added L2 ridge (`1e-12 * np.eye(N)`) to the inner QP
   Hessian when `N > P`. This makes the inner QP **strictly convex** (unique
   `W` for a given `V`), so the outer `V` trajectory is OS-deterministic for
   the cases where the minimizer is unique. **Covariate balance unchanged**
   (`max_w > 1e-2` still holds).
2. `833151a` — for `N > P`, run the inner QP from **2 starting points** and keep
   the better objective (`_solve_w` multi-start). This reduces the CI-observed
   `cov_mm` for the rank-deficient fixture from `0.327` → `0.0357` (**~9x**
   improvement) — but it is still **3.5x over the `1e-2` target**, because the
   underlying minimizer is non-unique and the chosen `W` differs across BLAS
   backends (CI's `scipy-openblas` vs local BLAS).

### Why it is xfail-gated rather than "fixed"

- The residual divergence (`cov_mm = 3.57e-02` on CI run `29250248678`, Py3.13)
  vs `~1e-8` on WSL is **deterministic per BLAS backend** (not run-to-run
  noise), rooted in the ill-conditioned (~`1e12`) ridge Hessian feeding SLSQP
  on CI's BLAS. It is a property of a **non-unique** optimization objective, not
  a bug we can close without changing the math.
- **Lead decision (final):** stop algorithmic escalation on Component 1. The
  assertion in `test_synth_rank_deficient_qp_same_objective_different_w`
  (`tests/test_synth.py:530`) is now `@pytest.mark.xfail(strict=False,
  reason="...")`. `strict=False` so a local BLAS match **xpasses** (reported,
  not failed) rather than erroring. The test's real purpose — `W` legitimately
  differs across optima yet `max_w > 1e-2` — still executes.
- **`synth()`'s validated R/Stata parity is unaffected** by this divergence;
  the xfail only concerns the cross-BLAS *agreement* assertion for the
  rank-deficient edge case.
- **Do not re-litigate or further escalate the Component 1 solver.** This is a
  deliberate, documented decision.

### Validation (Component 1)

- Windows (local): rank-deficient case bit-identical across 10 consecutive runs
  (`cov_mm` std = 0.0).
- Well-determined regression (`P >= N`): no ridge added; `max |W_diff| = 0.0`
  vs pre-fix.
- CI Linux: rank-deficient agreement test **xfails** (expected) on Py3.12/3.13;
  not a failure.

## Component 2 — placebo parity: validated green

`test_placebo_space_parity_r` (`tests/test_synth_placebo.py:~341`) validates the
donor-exclusion fix. Donor exclusions: `_MULTIMODAL_DONORS = {"d0","d1","d5","d7"}`,
threshold `5.0`; the `d2` coverage-gap flag is carried forward (NOT excluded).
This test ran GREEN on CI run `29250248678`.

**IMPORTANT COVERAGE CHANGE:** this test is now in the `slow` set (see CI
section below), so it **no longer runs on every push/PR** — only on the nightly
cron and manual `workflow_dispatch`. This is a real reduction in per-PR coverage
of the Component 2 fix, accepted in exchange for ~3-min-fast PR gating.

---

## CI restructuring (this branch)

Two workflows now exist:

- **`ci.yml`** — runs on `push` (main + this branch) and `pull_request` (main).
  **Single gate:** `pytest -m "not slow" tests/ ...` (fast suite only, both
  Py3.12/3.13). No slow step on the push/PR path.
- **`ci-slow.yml`** — runs on `schedule` (nightly `17 3 * * *` UTC) and
  `workflow_dispatch` (manual). **Not** on push/pull_request. Runs
  `pytest -m "slow" tests/ ...` (the 4 synth placebo sweeps).

### Why the split

The fast suite (689 tests) includes the re-enabled synth cross-OS tests, which
keep push/PR feedback reasonable. The 4 `slow`-marked tests are the
multi-method V-optimization (`_optimize_v`, `8b439d4`) placebo sweeps whose
local runtimes are 147s / 106s / 74s / 44s (~371s total). Under the old config
(both steps on every push) the branch run was ~22 min; removing the slow step
from the push/PR path restores the fast gate.

### Verified

- Fast push run `29254482804` (new config): **completed / success**, ran ONLY
  the fast suite (no slow step present). Wall-clock **9m39s** (created
  13:37:22Z → done 13:47:01Z).
- `slow` partition verified locally: `-m "not slow"` collects 689 (deselects
  4); `-m slow` collects exactly those 4.
- `pyproject.toml` `slow` marker registration and `addopts` are unchanged (bare
  local `pytest` still runs everything).

### Caveats / open points

- **Duration is ~9.5 min, not the ~3-min `main` baseline.** The gap is dominated
  by cold `pip install -e ".[dev,lint,plot]"` + 689-test execution, **not** the
  removed slow tests. The `main` 3-min figure is for `main`'s (smaller,
  synth-tests-disabled) suite, so it is not apples-to-apples with this branch's
  fast set. If a tighter wall-clock is required, add pip caching
  (`actions/setup-python` cache or `actions/cache`) and/or review whether more
  tests belong in the `slow` set.
- **`ci-slow.yml` cannot be manually dispatched from this feature branch.** The
  GitHub dispatch/schedule API resolves workflows from the **default branch
  (`main`)**, so `gh workflow run ci-slow.yml` returns 404 until the file is
  merged to `main`. The nightly cron + manual dispatch will become live once
  PR #19 is merged. This is a GitHub limitation, not a config error — the YAML
  is valid and the trigger config is correct.
- Stale pre-restructure runs (`29253321858`, `29253324524`, old both-step
  config) may still be finishing on CI; they are harmless and can be ignored or
  cancelled.

---

## Files Changed (this branch, full chain)

| File | Change |
|------|--------|
| `open_econs/models/causal/synth.py` | L2 ridge in `_solve_w()` when `N > P`; multi-start inner QP; multi-method V-optimization (`_optimize_v`) |
| `tests/test_synth.py` | Rank-deficient test xfail-gated (`:530`) |
| `tests/test_synth_placebo.py` | Donor-exclusion logic (Component 2); 4 tests marked `@pytest.mark.slow` |
| `pyproject.toml` | `slow` marker registered; `addopts` unchanged |
| `.github/workflows/ci.yml` | Fast-only gate on push/PR |
| `.github/workflows/ci-slow.yml` | NEW — nightly + manual slow safety net (no push/PR) |
| `docs/synth-cross-os-solver-recon-update.md` | This file (corrected from the false "Status: Fixed") |

## Still Open (unchanged)

- `nlogit()` — recon complete (`docs/nlogit-recon.md`); implementation blocked on
  `mlogit`-spec equality-structure syntax.
- API freeze candidate — no more breaking signature changes without deprecation.
- Full docstring coverage + type-checked public API.
- GEE — candidate for v1.0+ if demand materialises.
- (Scoped, not built) runtime `UserWarning` for rank-deficient / multi-modal
  donor pools, mirroring `nls()`.
