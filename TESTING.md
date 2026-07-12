# Testing

open-econs has a large Stata-/R-backed **parity** test suite: tests that
compare estimator output against genuine Stata (`.dta` fixtures) or R
(`Rscript`) ground truth. Running those binaries on every local iteration is
slow and, historically, even regenerated the committed `.dta` fixtures as a
side effect.

The suite is therefore tiered with pytest markers and an explicit fixture-
regeneration gate. **A genuine parity check against Stata/R output is still
required before every merge** — this only changes *when* and *how easily* that
check runs, not whether it exists.

## Markers

| Marker | Meaning |
|--------|---------|
| `@pytest.mark.stata` | Compares against a Stata-generated `.dta` fixture. |
| `@pytest.mark.r` | Compares against R output produced via `Rscript`. |

Both markers are **excluded by default** via `addopts` in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = '-m "not stata and not r" --ignore=tests/stata/do/archive'
```

So a bare `pytest` never collects these tests.

## Three invocation modes

### 1. Default — fast, no Stata/R, no fixture writes (routine iteration)

```bash
pytest            # or: pytest tests/
```

- Skips every `stata`/`r` test (deselected by the default `-m` expression).
- Reads committed `.dta` fixtures only where modules import them at load time;
  it never launches Stata or R and never modifies a fixture file on disk.
- This is the mode for everyday local work.

### 2. Parity check — read committed fixtures, compare against ground truth

```bash
pytest -m "stata or r"
```

- Runs the gated parity tests. A CLI `-m` overrides the default `-m` in
  `addopts`, so this selects exactly the `stata`/`r` tests.
- Each test reads the **already-committed** `.dta` (or invokes `Rscript` if R
  is installed) and runs the same strict numeric comparison as before — no
  tolerance or assertion logic is weakened.
- Use this before opening a PR to confirm parity claims still hold.
- Fixtures are **read, not regenerated**: no `.dta` file is rewritten.

### 3. Fixture regeneration — full live `.do`/R run (rare)

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

CI (`.github/workflows/ci.yml`) has no Stata/R binaries, so the parity tests
run in committed-fixture fallback mode. It runs **two** invocations to preserve
the full pre-change test count:

1. `pytest tests/ ...` — the default run (excludes `stata`/`r`).
2. `pytest -m "stata or r" tests/ ...` — the parity run (reads committed
   fixtures; tests needing a binary skip cleanly via their existing
   `skipif` gates).

The drift check in `stata_runner.py` still fails a parity test loudly if a
`.do` file is newer than its committed `.dta`, catching stale fixtures.
