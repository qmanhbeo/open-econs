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

The `stata`/`r` markers are **no longer excluded by default**. `addopts` in
`pyproject.toml` keeps only the `tests/stata/do/archive` ignore; a bare
`pytest` collects and runs every test, including the parity tests:

```toml
[tool.pytest.ini_options]
addopts = '--ignore=tests/stata/do/archive'
```

`-m "not stata and not r"` remains available as an explicit opt-in *fast mode*
that deselects the parity tests.

## Invocation modes

### 1. Default — full suite, fixture-only (routine iteration)

```bash
pytest            # or: pytest tests/
```

- Runs **every** test in the suite, including the `stata`/`r` parity tests.
- All parity tests read committed `.dta` fixtures only; `run_do()` is gated
  behind `OE_REGENERATE_FIXTURES`, so no Stata or R binary is launched and no
  fixture file is rewritten on disk.
- This is the mode for everyday local work — it now exercises the full suite,
  so parity regressions surface without a separate command.

### 2. Fast mode — skip parity tests (quick local iteration)

```bash
pytest -m "not stata and not r"
```

- Deselects every `stata`/`r` test, so only the fast, non-parity tests run.
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
- Only needed when you changed a `.do` (or R) script, or right before a
  release. R-backed tests invoke `Rscript` directly (their outputs land in
  temp dirs, not in the repo).
- After regenerating, **revert any fixture you did not intend to change**
  (`git checkout -- tests/stata/do/*.dta`) so regenerated fixtures are never
  left uncommitted. Commit the `.do` and its regenerated `.dta` together.

## CI

CI (`.github/workflows/ci.yml`) has no Stata/R binaries. The single
`pytest tests/ ...` step runs the **full suite** (607 tests): because the
default run includes the `stata`/`r` parity tests and they read committed
fixtures only, no binary is launched. The separate `-m "stata or r"` parity
step was removed as redundant — it only re-ran the 216-test parity subset
already covered by the full default run. The drift check in `stata_runner.py`
still fails a parity test loudly if a `.do` file is newer than its committed
`.dta`, catching stale fixtures.
