# open-econs v1.4.2 — Release Notes

**Status: PUBLISHED.** Version bumped to `1.4.2` in `pyproject.toml` and
`open_econs/_version.py`. Tagged `v1.4.2`; the GitHub Release `published`
event triggers `publish.yml` (PyPI) and `ci-parity.yml`.

## Summary

Patch release closing the **ROBUST-REG-STATA** open parity gap. No API change.

- `robust_reg(parity='stata')` now reproduces Stata `rreg` coefficients
  (`e(b)`) and robust SEs (`e(V)`) to **<3e-10** (machine precision).
- Root cause: Stata's `rreg` reports coefficients from a **final
  bias-correction regression** (not the last biweight step) and uses
  effective N (non-zero-weight observations) in the correction's `lambda`.
- The R-backed `parity='rlm'` branch is unchanged (already 1e-6).
- The strict coef/SE assertions in `tests/stata/tests/test_stata_rreg.py`
  now pass (xfail removed). Root cause documented in
  `methodology/linear/robust_reg.md`.

## Full changelog

See `CHANGELOG.md` (`[1.4.2]` entry).
