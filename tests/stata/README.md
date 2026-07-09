# Stata Parity Tests

Compare open-econs results against StataMP output to validate numerical correctness.

## Architecture

```
tests/stata/
  fixtures/          Fixed CSV datasets (committed to repo)
  do/                Hand-written Stata .do files (committed to repo)
  stata_runner.py    Subprocess wrapper for calling StataMP
  conftest.py        Pytest fixtures that load the fixed CSVs
  test_stata_*.py    Parity tests
```

**Flow:**
1. Fixed CSV datasets in `fixtures/` are imported by both Python and Stata
2. Stata `.do` files import CSVs, run commands, export `.dta` result files
3. Python tests import the same CSVs, run open-econs, compare against `.dta`

## Prerequisites

- StataMP 17 installed at `C:\Program Files\Stata17\StataMP-64.exe`
- Override path with env var `STATA_EXE` if different
- SSC packages: `oaxaca`, `xtabond2`, `rdrobust`, `csdid`, `drdid`

## Running

```bash
# Run all parity tests
python -m pytest tests/stata/ -v

# Run only non-SSC tests (no Stata packages needed)
python -m pytest tests/stata/ -v -k "not oaxaca and not abond and not rdd and not staggered"
```

## Test Status

| Module | Tests | Status |
|--------|-------|--------|
| OLS (basic, robust, cluster, HAC, predict, confint) | 12 | All pass |
| Panel (FE, RE, pooled, FD, Hausman) | 14 | Stata correct, open-econs API needs adaptation |
| IV / 2SLS | 3 | Stata correct, open-econs formula needs fixing |
| Logit / Probit | 10 | Stata correct, open-econs API needs adaptation |
| DiD (basic, cluster) | 3 | Pass |
| Event Study | 1 | Stata correct, open-econs needs `treat_event_time` column |
| Balance | 2 | Stata correct, open-econs returns different column names |
| Oaxaca (two-fold, three-fold) | 3 | Stata correct, open-econs uses different decomposition method |
| Arellano-Bond | 2 | Stata correct, open-econs returns different coefficient count |
| RDD (sharp, fuzzy) | 5 | Stata correct, different bandwidth algorithm = different point estimates |
| Staggered DiD | 1 | Stata correct, open-econs API needs fixing |

**17 pass, 29 fail from open-econs API mismatches (not Stata issues).**

## Notes

- All `.do` files must be ASCII/UTF-8 **without BOM** — Stata cannot parse BOM
- Stata `xtreg, fd` is not valid in Stata 17 — use manual first-differencing
- Stata `oaxaca` two-fold uses `pooled` or `omega`, not `two-fold`
- Stata `csdid` `gvar` must be the treatment year (e.g., 0, 3, 5), not binary 0/1
