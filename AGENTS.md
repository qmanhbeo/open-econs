# AGENTS.md — Standard of Practice

Binding rules for all agents working in this repository. The overriding
principle: **parity with Stata and R is the product.** Every option of an
estimator must reproduce its reference implementation to a known tolerance, or
be explicitly flagged and preserved as a documented divergence. When in doubt,
stop and report.

## 1. Verify against source, not output
Before claiming a result "matches" Stata/R, confirm the convention against the
reference implementation (Stata `.ado`/Mata source, R package source, or a
live extracted quantity such as `e(S)`, `e(Wmatrix)`, `e(b)`, or R's
`summary()`/coef output). Output-only matching hides silent convention
differences. If a package is not loadable in the local R install, regenerate
fixtures through the test harness, not a bare Rscript probe.

## 2. Parity is the deliverable — never loosen tolerance
The maximum tolerance for numeric parity is **1e-6** (rule 2). Never widen a
tolerance to paper over a discrepancy. If a gap exists, diagnose to root cause
and either fix it or flag it as a documented, guarded divergence. Dropping or
weakening a crosscheck is itself a regression.

## 3. Optionality is a feature, not a nuisance
Every estimator option must be audited. If the reference behavior cannot be
reproduced, expose the disparity as a **toggle** and cover BOTH settings with
tests (rule 15). A convention divergence is an open item, not a reason to drop
the option.

## 4. One concern per commit
Each commit addresses exactly one logical change (fix, test, doc, fixture).
Coherent but separable. Never bundle an unrelated refactor with a behavior fix.

## 5. Exclude `synth_placebo`
Always run the test suite with `-m "not synth_placebo"` unless explicitly
working on the placebo tests themselves.

## 6. Stop and report on anything suspicious
If a number looks wrong, a convention contradicts the docs, a test passes for
the wrong reason, or a regeneration silently changed a fixture — stop. Report
before proceeding. No silent "fixes."

## 7. Standardized test layout
Parity tests live under `tests/stata/tests/` and `tests/r/tests/` in files
named after the estimator (e.g. `test_stata_gmm.py`, `test_r_gmm.py`). Shared
fixtures are committed (`.dta`/`.json`) and regenerated only with
`OE_REGENERATE_FIXTURES=1`. Read with
`tests/stata/stata_runner.py::read_stata(label)` and
`tests/r/r_runner.py::read_r(label)`.

## 8. Cache shared fixtures
Do not regenerate fixtures on every run. Commit the generated artifacts; gate
regeneration behind `OE_REGENERATE_FIXTURES=1`. The drift check fails if a
generator (`.do`/`.R`) is newer than its committed artifact.

## 9. Use subagents for separated work
When a task splits into independent, mechanical pieces (e.g. several new test
classes, fixture regeneration, doc updates), delegate with bounded prompts
rather than doing everything inline.

## 10. Gate before pushing
Before any `git push`, run the full gate: `ruff check`, `mypy`, and
`pytest -m "not synth_placebo"`. Do not push a red gate.

## 11. Bounded prompts
Keep subagent and tool prompts tightly scoped with explicit return contracts.
Avoid open-ended "explore the codebase" instructions.

## 12. Exhausted work goes to FUTURE_WORK
When an audit is done or a path is blocked, append the status to the bottom of
`FUTURE_WORK.md` (or update the matching section). Do not leave loose TODOs in
chat only.

## 13. Check FUTURE_WORK before exploring
Before investigating "why X differs," grep `FUTURE_WORK.md` and `methodology/`.
The root cause may already be recorded (rule 16).

## 14. Prefer wrapping if machine-precision, else read source
If a convention is genuinely a user-facing choice (e.g. two valid J-stat
definitions), expose it as a parameter rather than hardcoding one. Otherwise
read the reference source to pick the correct default.

## 15. Expose disparities as toggles + cover both settings
When reference implementations disagree (Stata vs R, or literature vs software),
add a parameter that selects the convention and write a test for each branch.
See `gmm()`'s `windmeijer`, `robust_meat`, `weight`, and `hac_weighting`.

## 16. Record root causes in `methodology/`
Any non-trivial convention decision or bug fix gets a note in
`methodology/<area>/<model>.md`. Future agents must not have to re-derive it
(rule 13 relies on this).

## 17. Move diagnosis files to `archive/`
Scratch diagnostic scripts/data (e.g. `gmm_diag.do`) go under
`<area>/generate-fixtures/archive/` so they are available if re-tracing is ever
needed, but do not clutter the active generator set.

## 18. No footguns — flag them
If a parameter's semantics are subtle (e.g. `robust_meat` switches only the
meat `S2`, not the bread), document the footgun inline AND in the methodology
note. A future "simplification" must not silently regress parity.
