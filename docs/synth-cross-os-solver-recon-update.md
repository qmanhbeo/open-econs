# `synth()` — Cross-OS Solver Nondeterminism: Status Update

**Status:** Partially mitigated — Component 1 (rank-deficient inner QP) is
**xfail-gated, not algorithmically fixed**; Component 2 (placebo parity) is
**validated green**. CI restructured so push/PR runs only the fast suite (now
**~2.5 min local**, wall-clock targeted to ~3–4 min via pip/mypy caching) with
the slow synth sweeps moved to a nightly + manual safety net.
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

- **Duration ~9.5 min on the pre-caching run, not the ~3-min `main` baseline.**
  The gap was dominated by cold `pip install -e ".[dev,lint,plot]"` + `mypy`
  (no cache) overhead, **not** the removed slow tests and not the fast-suite
  tests themselves (the fast suite runs in ~2.5 min locally). The `main` 3-min
  figure is for `main`'s smaller, synth-tests-disabled suite, so it is not
  apples-to-apples.
- **Pip + mypy caching added (2026-07-13).** `ci.yml` now uses
  `actions/setup-python` `cache: pip` (keyed on `pyproject.toml`) and an
  `actions/cache` on `.mypy_cache` keyed on
  `mypy-<python-version>-<hash(pyproject.toml)>`. This is expected to close most
  of the 9.5-min → ~3–4-min gap on **warm-cache** runs. The **first run after
  this change is still cold** (no cache yet) and will be slow — that is expected,
  not a regression. See "Test tiering follow-up" for the test-side changes too.
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

## Test tiering follow-up (2026-07-13)

Durations probe (`pytest -m "not slow" --durations=50`) showed the fast suite
executes in **214.58s (3:34)** locally — already near the ~3-min target — so
the 9.5-min CI wall-clock is CI-job overhead (cold pip + mypy + coverage), not
the tests. The following surgical re-tiering/optimization was applied to the
fast set's slowest outliers; **no test was skipped or deleted** ("move" = runs
on the nightly/manual `ci-slow.yml` path only).

### `test_placebo_immutability` → moved to `slow`
- 18.66s → now `slow`. It only asserted pydantic-v2 `BaseModel` immutability
  (framework behaviour), already covered by `test_synth_immutability`
  (`SynthResult` is the same `BaseModel` family). Zero project-logic coverage
  loss.

### `test_placebo_plot_predict_stubs` + `test_synth_predict_plot_stubs` → optimized (kept fast)
- Both now build a **lightweight result object directly** (minimal
  `PlaceboSpaceResult` / `SynthResult`, no `synth()`/`placebo_space()` fit) and
  still assert `plot()`/`predict()` raise `NotImplementedError`.
- **Plot-deprecation finding (confirmed for synth/placebo results):** `plot()`
  and `predict()` are deliberate `NotImplementedError` stubs on every synth
  result object (`SynthResult`/`PlaceboSpaceResult`/`PlaceboTimeResult` —
  ROADMAP.md, `synth.py:846`, `placebo.py:79/193`), with `pd.DataFrame`/pydantic
  outputs as the supported interface. NOTE: plotting is **not** deprecated
  project-wide — `OLSResult.plot()` is deprecated (v0.8) and `DiD`-style results
  implement `plot()`, but the synth/placebo stub contract is exactly what these
  two tests guard. The optimization is therefore safe.

### `test_placebo_space_internal_consistency` + `test_placebo_space_returns_placebo_result` → optimized (kept fast)
- 19.77s / 18.60s → both now use a **smaller panel** (`_make_panel_small`: 6
  donors / 12 periods vs 12 / 20), exercising the full `placebo_space` path
  end-to-end but refitting for far fewer donors. Correctness-critical (no faster
  equivalent) so kept in the fast gate rather than moved.

### Result
- Fast suite: **144.55s (2:24)** after changes (was 3:34). 657 passed, 19
  pre-existing NLS/sympy `ImportError` failures (unrelated), 11 skipped, **5
  deselected** (the 4 original `slow` + `test_placebo_immutability`), 1 xpassed.
- **Open follow-up (out of scope, flagged):** `test_placebo_time_returns_placebo_result`
  is now the fast set's slowest at ~29s — and its runtime is *variable* (it
  refits `synth()` per pre-period candidate; the multi-method `V`-optimization
  has run-to-run solver variance). It was **not** in the re-tiering list and was
  left untouched; a future pass could point it at `_make_panel_small` too.

---

---

## Files Changed (this branch, full chain)

| File | Change |
|------|--------|
| `open_econs/models/causal/synth.py` | L2 ridge in `_solve_w()` when `N > P`; multi-start inner QP; multi-method V-optimization (`_optimize_v`) |
| `tests/test_synth.py` | Rank-deficient test xfail-gated (`:530`); `test_synth_predict_plot_stubs` optimized via lightweight `SynthResult` |
| `tests/test_synth_placebo.py` | Donor-exclusion logic (Component 2); 4 tests marked `@pytest.mark.slow`; `test_placebo_immutability` moved to `slow`; plot-stub + 2 consistency/shape tests optimized (smaller panel / lightweight `PlaceboSpaceResult`) |
| `pyproject.toml` | `slow` marker registered; `addopts` unchanged |
| `.github/workflows/ci.yml` | Fast-only gate on push/PR; **pip cache (`setup-python`) + `.mypy_cache` cache added** |
| `.github/workflows/ci-slow.yml` | NEW — nightly + manual slow safety net (no push/PR) |
| `docs/synth-cross-os-solver-recon-update.md` | This file (corrected from the false "Status: Fixed"; caching + tiering follow-up) |

## Still Open (unchanged)

- `nlogit()` — recon complete (`docs/nlogit-recon.md`); implementation blocked on
  `mlogit`-spec equality-structure syntax.
- API freeze candidate — no more breaking signature changes without deprecation.
- Full docstring coverage + type-checked public API.
- GEE — candidate for v1.0+ if demand materialises.
- (Scoped, not built) runtime `UserWarning` for rank-deficient / multi-modal
  donor pools, mirroring `nls()`.
