# `synth()` — Cross-OS Solver Nondeterminism: Status Update

**Status:** Partially mitigated — Component 1 (rank-deficient inner QP) is
**xfail-gated, not algorithmically fixed**; Component 2 (placebo parity) is
**validated green**. CI restructured so push/PR runs only the fast suite; the
slow synth sweeps moved to a nightly + manual safety net. Per-PR wall-clock is
now **~4–5 min** (was ~6–8 min): the fast suite itself measures **~263s on real
CI hardware** (both Py3.12/3.13, no coverage), and the earlier ~3–4 min target
was missed because **`--cov` on Python 3.13 adds ~60% overhead** (~157s) — now
fixed by running 3.13 without `--cov` (3.12 still gathers coverage). See
"CI duration — measured on real hardware".
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
of the Component 2 fix, accepted in exchange for fast PR gating.

**SECOND COVERAGE CHANGE (Py3.13 per-PR):** the push/PR fast gate no longer
passes `--cov` on the Python 3.13 leg (it still does on 3.12), because coverage.py
on 3.13 (PEP 669 `sys.monitoring`) adds ~60% test-step overhead vs ~6% on 3.12.
Disclosed here per the project's honesty standard: 3.13-only code paths lose
**live** coverage reporting on every PR (coverage is still gathered on 3.12,
still available locally, and the `slow` suite still runs on 3.13 nightly). It is
a coverage-*reporting* tradeoff for speed, not a reduction in test execution.

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

> **CORRECTION (supersedes the original draft).** The earlier claim that the
> ~9.5-min CI wall-clock was "dominated by cold `pip install` + `mypy` (no cache)
> overhead, **not** the fast-suite tests" was **wrong**. Real per-test CI
> measurement (run `29259559738`) showed the `pytest` step alone was **6m57s of a
> 7m44s run** — the tests, not install/type-check, dominate. Raw (no-coverage)
> fast suite is **~263s on both Py3.12 and Py3.13**; the extra ~3–4 min came from
> (a) CI runners having fewer/shared vCPUs vs local (~1.8x) and (b) `--cov` on
> Python 3.13 specifically (+60%, ~157s). Pip/mypy caching was added and verified
> working, but it was a minor contributor, **not** the fix.

- **Per-PR wall-clock is now ~4–5 min** (down from ~6–8 min). See "CI duration —
  measured on real hardware" (below) for the breakdown and the applied fixes.
- The `main` 3-min figure is for `main`'s smaller, synth-tests-disabled suite, so
  it is not apples-to-apples with this branch's ~4–5 min.
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

### CI-measured durations (real hardware — supersedes the local probe above)

A temporary `ci-durations-diag.yml` (since removed in `9671521`) captured per-test
timings on real CI runners (see run `29259559738`, Py3.12/3.13 × with/without
`--cov`). The local probe is
**not** a substitute for these numbers. Headlines:

| Leg | no-cov total | with-cov total | `--cov` overhead |
|---|---|---|---|
| Py 3.12 | 265.4s | 281.1s | +15.7s (+5.9%) |
| Py 3.13 | 259.9s | 416.8s | +156.8s (+60.3%) |

- Raw (no-coverage) fast suite ≈ **263s on BOTH legs** — version-independent.
- **The cost is `--cov` on Python 3.13: +60% vs only +6% on 3.12** (coverage.py's
  PEP 669 `sys.monitoring` path). This — not cold install/mypy — is why per-PR hit
  6–8 min.
- Elbow: top 11 tests (1.6%) = 80% of runtime (threshold ≈10.5s); top 6 = 50%.
- Heavy tail: `test_placebo_time_returns_placebo_result` (~46s); a cluster of six
  `SynthResult` contract tests each ~18s (each re-fits `synth()` independently
  rather than sharing a fit); `test_gmm::test_hansen_j_size_and_power` (~14s).

### Second follow-up (2026-07-13) — applied after CI measurement

1. **Shared module-scoped `synth()` fixture** (`tests/test_synth.py`) for the six
   `SynthResult` contract tests (returns / immutability / tidy+export / vcov stub /
   default-predictors / ground-truth-recovery / validation-missing-columns).
   Collapses 6×~18s → ~18s total. No test moved, no coverage lost.
   (`test_synth_immutability` was also moved to `slow` — see below — and uses the
   shared fit.)
2. **`test_synth_immutability` → moved to `slow`** (its twin `test_placebo_immutability`
   already was; both only check pydantic-v2 `BaseModel` immutability — framework
   behaviour, not project logic).
3. **`test_placebo_time_returns_placebo_result` → smaller panel** (`_make_panel_small`,
   6 donors/12 periods vs 12/20), cutting ~46s → ~10s. Exercises the full
   `placebo_time` path; correct, just cheaper.
4. **`test_gmm::test_hansen_j_size_and_power` → `R` 500 → 300** (seeded MC, so
   deterministic). Size tolerance widened to `[0.015, 0.11]` to match R=300 s.e.;
   power check unchanged. Unrelated to synth; flagged repeatedly as borderline.
5. **`--cov` dropped on the Py3.13 per-PR leg** (kept on 3.12). Disclosed above and
   in `ci.yml`. Expectation: 3.13 test step 414.8s → ~259.9s.

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

### Result (first follow-up — local probe only)

- Fast suite: **144.55s (2:24)** after the first re-tiering (was 3:34) — **local
  only**; the CI-measured figure (~263s, no coverage) supersedes it for
  decision-making (CI runners have fewer/shared vCPUs, ~1.8x).
- 657 passed, 19 pre-existing NLS/sympy `ImportError` failures (unrelated), 11
  skipped, **5 deselected** (the 4 original `slow` + `test_placebo_immutability`),
  1 xpassed.

### Result (after second follow-up — CONFIRMED by push run `29262027059`)

- `SynthResult` cluster: 6×~18s → ~18s (shared fixture).
- `test_placebo_time_returns_placebo_result`: ~46s → ~10s (smaller panel).
- `test_gmm::test_hansen_j_size_and_power`: ~14s → ~9s (R 300).
- Py3.13 test step: ~414.8s (with `--cov`) → ~259.9s no-coverage, and the per-PR
  leg now RUNS WITHOUT `--cov`, so the 3.13 job finished in **2m1s** and the 3.12
  job (still with `--cov`) in **2m48s**. Per-PR feedback is now ~2–3 min/leg,
  comfortably under the original ~3–4 min target (and far below the prior 6–8 min).
- Source of truth: push run `29262027059` (both legs green).

---

---

## Files Changed (this branch, full chain)

| File | Change |
|------|--------|
| `open_econs/models/causal/synth.py` | L2 ridge in `_solve_w()` when `N > P`; multi-start inner QP; multi-method V-optimization (`_optimize_v`) |
| `tests/test_synth.py` | Rank-deficient test xfail-gated (`:530`); `test_synth_predict_plot_stubs` optimized via lightweight `SynthResult`; **shared module-scoped `synth()` fixture for the 6 `SynthResult` contract tests**; `test_synth_immutability` moved to `slow` |
| `tests/test_synth_placebo.py` | Donor-exclusion logic (Component 2); 4 tests marked `@pytest.mark.slow`; `test_placebo_immutability` moved to `slow`; plot-stub + 2 consistency/shape tests optimized; `test_placebo_time_returns_placebo_result` switched to `_make_panel_small` |
| `tests/test_gmm.py` | `test_hansen_j_size_and_power`: `R` 500 → 300 (size tolerance widened to match) |
| `pyproject.toml` | `slow` marker registered; `addopts` unchanged |
| `.github/workflows/ci.yml` | Fast-only gate on push/PR; pip cache (`setup-python`) + `.mypy_cache` cache; **`--cov` now conditional — dropped on Py3.13 leg, kept on 3.12** |
| `.github/workflows/ci-slow.yml` | NEW — nightly + manual slow safety net (no push/PR) |
| `.github/workflows/ci-durations-diag.yml` | TEMPORARY diagnostic — REMOVED in `9671521` (captured run `29259559738`; per-push cost no longer justified) |
| `docs/synth-cross-os-solver-recon-update.md` | This file (corrected false "Status: Fixed"; caching + tiering follow-up + CI-measured durations + second follow-up) |

## Still Open

- **CI: `ci-durations-diag.yml` REMOVED** in `9671521`. The temporary diagnostic
  workflow (push trigger) was deleted once run `29259559738` had captured the
  per-test data it was added for; its per-push CI cost was no longer justified.
- `nlogit()` — recon complete (`docs/nlogit-recon.md`); implementation blocked on
  `mlogit`-spec equality-structure syntax.
- API freeze candidate — no more breaking signature changes without deprecation.
- Full docstring coverage + type-checked public API.
- GEE — candidate for v1.0+ if demand materialises.
- (Scoped, not built) runtime `UserWarning` for rank-deficient / multi-modal
  donor pools, mirroring `nls()`.
