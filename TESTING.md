# Testing

open-econs has a large Stata-/R-backed **parity** test suite: tests that
compare estimator output against genuine Stata (`.dta` fixtures) or R
(`Rscript`) ground truth. Running those binaries on every local iteration is
slow and, historically, even regenerated the committed `.dta` fixtures as a
side effect.

The suite uses pytest markers plus an explicit fixture-regeneration gate.
**A genuine parity check against Stata/R output is still required before every
merge** — and it now happens automatically in the default `pytest` run, which
executes the full suite (including `stata`/`r` parity tests) against committed
fixtures. The `-m "not stata and not r"` expression is kept only as an opt-in
*fast mode* for quick local iteration that skips the parity tests.

## Markers

| Marker | Meaning |
|--------|---------|
| `@pytest.mark.stata` | Compares against a Stata-generated `.dta` fixture. |
| `@pytest.mark.r` | Compares against R output produced via `Rscript`. |
| `@pytest.mark.synth_placebo` | Synthetic control and placebo inference tests. Excluded from default runs. |

### Parity scope and counts

Collected totals (verified with `pytest --collect-only`):

- **Full suite:** 705 tests.
- **Parity subset (`-m "stata or r"`):** 217 tests = **208 Stata** + **9 R**,
  with **zero** tests marked for both engines (the `stata and r` intersection is
  empty). The 208/9 split counts test *functions*, not methods.

The `stata` and `r` suites are **not** a clean partition of methods:

- The 9 R tests mostly cover estimators that have **no** Stata fixture:
  synthetic control (`test_synth.py`), NLS (`test_nls.py`), FE-HAC
  (`test_fe_hac.py`), and synthetic-control placebo (`test_synth_placebo.py`).
- A few methods carry parity against **both** engines via separate tests — the
  example is `mlogit` (`tests/stata/test_mlogit.py` has 3 Stata-marked tests and
  1 R-marked test).

### Tolerance

Parity assertions are not all to "machine precision." Tolerances run from
`1e-6` (the common default, used in ~80 checks) up to `1e-15` for the tightest
exact-equivalence cases (e.g. IV, Arellano-Bond, synthetic control), with most
checks clustered at `1e-6`–`1e-10`. Refer to individual test files for the
exact `rtol`/`atol` on each comparison.

The `stata`/`r` markers are **no longer excluded by default**. `addopts` in
`pyproject.toml` keeps only the `tests/stata/generate-fixtures/archive` ignore; a bare
`pytest` collects and runs every test, including the parity tests:

```toml
[tool.pytest.ini_options]
addopts = '--ignore=tests/stata/generate-fixtures/archive'
```

`-m "not stata and not r"` remains available as an explicit opt-in *fast mode*
that deselects the parity tests.

## Invocation modes

### 1. Default — full suite, fixture-only (routine iteration)

```bash
pytest -m "not synth_placebo"            # or: pytest -m "not synth_placebo" tests/
```

- Runs every test **except** synth-placebo tests, including the `stata`/`r` parity tests.
- All parity tests read committed `.dta` fixtures only; `run_do()` is gated
  behind `OE_REGENERATE_FIXTURES`, so no Stata or R binary is launched and no
  fixture file is rewritten on disk.
- This is the mode for everyday local work — it exercises the full suite,
  so parity regressions surface without a separate command.

### 2. Fast mode — skip parity and synth-placebo tests (quick local iteration)

```bash
pytest -m "not stata and not r and not synth_placebo"
```

- Deselects every `stata`/`r` test and every `synth_placebo` test, so only the
  fast, non-parity tests run.
- Same fixture-only behaviour as the default; just a smaller, quicker subset.
- Use this when iterating on non-parity code and you don't need the parity
  assertions on every run.

### 3. Parity-only selection — read committed fixtures, compare against ground truth

```bash
pytest -m "stata or r"
```

- Selects **only** the `stata`/`r` parity tests (the inverse of the fast mode).
- Each test reads the **already-committed** `.dta` (or invokes `Rscript` if R
  is installed) and runs the same strict numeric comparison as the default
  run — no tolerance or assertion logic is weakened.
- Fixtures are **read, not regenerated** unless `OE_REGENERATE_FIXTURES` is set
  (see mode 4): no `.dta` file is rewritten.

### 4. Fixture regeneration — full live `.do`/R run (rare)

```bash
OE_REGENERATE_FIXTURES=1 pytest -m "stata or r"
```

- The `OE_REGENERATE_FIXTURES` environment variable (any truthy value) opt-*in*
  gate lets `tests/stata/stata_runner.py:run_do()` actually launch StataMP and
  overwrite the `.dta` fixtures from their `.do` scripts.
- For **Stata**, `run_do()` launches StataMP and overwrites the committed
  `.dta` fixtures; for **R**, `run_r()` launches `Rscript` and overwrites the
  committed `tests/r/fixtures/<label>.json` fixtures (and, if the panel shape
  changed, the `<label>_input.csv`).  R outputs are **committed to the repo**,
  not written to temp dirs — they are the shared ground truth both the Python
  test and the `.R` script read (see `tests/r/README.md`).
- Only needed when you changed a `.do` (or `.R`) script, or right before a
  release.
- After regenerating, **revert any fixture you did not intend to change**
  (`git checkout -- tests/stata/generate-fixtures/*.dta tests/r/fixtures/*.json`) so regenerated
  fixtures are never left uncommitted. Commit the `.do`/`.R` and its regenerated
  fixture together.

## R parity fixtures (`tests/r/`)

The R-backed tests (`@pytest.mark.r`) follow the same dual-mode pattern as the
Stata suite, but with a twist: the **input panel is a single committed CSV
shared by both sides**. `tests/r/fixtures/<label>_input.csv` is read by the
Python test (`_panel_from_csv`) *and* by the `.R` script (`argv[1]`), so there
is no cross-engine RNG-sync assumption; the expected output
`tests/r/fixtures/<label>.json` is written by the `.R` script (`argv[2]`) during
regeneration and read by the Python test via `read_r(<label>)`. Both modes apply
the same `>` drift check (`.R` newer than `.json` ⇒ `STALE FIXTURE`).

Key maintainer notes (full detail in `tests/r/README.md`):

- Regeneration is gated behind `OE_REGENERATE_FIXTURES` and requires
  `R_EXE` (default `C:\Program Files\R\R-4.6.1\bin\Rscript.exe`) to point at a
  valid Rscript; otherwise the committed `.json` is read and no R is launched.
- R fixtures are **committed to the repo**, not written to temp dirs.
- The `synth` placebo-space test's `median_ratio` guard is `< 3.0` (not `< 1.0`)
  by design — it reflects the documented nonconvex-`V` / rank-deficient-donor
  divergence, not a `time`-dtype bug.  See `tests/r/README.md` for the trace.

## CI

CI (`.github/workflows/ci.yml`) has no Stata/R binaries. The single
`pytest tests/ ...` step runs the **full suite** (705 tests): because the
default run includes the `stata`/`r` parity tests and they read committed
fixtures only, no binary is launched. The separate `-m "stata or r"` parity
step was removed as redundant — it only re-ran the 217-test parity subset
(208 vs Stata, 9 vs R) already covered by the full default run. The drift check
in `stata_runner.py`
still fails a parity test loudly if a `.do` file is newer than its committed
`.dta`, catching stale fixtures.
