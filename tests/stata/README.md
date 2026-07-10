# Stata Parity Tests — Maintainer Guide

## Dual-mode execution

These tests run in two modes:

| Mode | When | Behaviour |
|------|------|-----------|
| **Live Stata** | `STATA_EXE` points to a valid StataMP binary | Runs every `.do` file, regenerates `.dta`, compares |
| **Committed fallback** | No StataMP (CI, contributors without Stata) | Reads committed `.dta` fixtures directly |

Both modes apply a **drift check**: if any `.do` file is newer than its `.dta`,
the test fails with `STALE FIXTURE`.  This catches the common mistake of editing
a `.do` but forgetting to regenerate the `.dta`.

### Regenerating fixtures

When you change a `.do` file, regenerate its `.dta`:

```bash
# On a machine with StataMP 17:
& "C:\Program Files\Stata17\StataMP-64.exe" /e do tests\stata\do\my_test.do

# Or via Python (will call StataMP if available):
python -c "from tests.stata.stata_runner import run_do; run_do('my_test')"
```

Then commit both the `.do` and the `.dta`.

### CI behaviour

On GitHub Actions (ubuntu-latest, no StataMP), `tests/stata/` runs in
committed-fallback mode.  The `.dta` fixtures are version-controlled so the
parity assertions execute without Stata.

---

## How to write a correct `.do` file

### 1. File encoding: NO BOM

Stata cannot parse UTF-8 BOM (`EF BB BF`). If your `.do` file starts with `﻿`, Stata
will error with `is not a valid command name`.

**Wrong** (Python `Path.write_text(encoding="utf-8")` adds BOM on Windows):
```python
Path("my_file.do").write_text(content, encoding="utf-8")  # BROKEN
```

**Correct**:
```python
Path("my_file.do").write_bytes(content.encode("ascii"))  # SAFE
```

Or manually strip BOM after writing:
```powershell
$bytes = [System.IO.File]::ReadAllBytes("file.do")
if ($bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
    [System.IO.File]::WriteAllBytes("file.do", $bytes[3..($bytes.Length-1)])
}
```

### 2. Export ALL coefficients (including intercept)

When open-econs returns 3 coefficients `[intercept, x, z]` and your `.do` exports 2
`[x, z]`, the test gets a shape mismatch.  **Always export the intercept.**

```stata
* WRONG — missing intercept
scalar s_bx = _b[x]
scalar s_bz = _b[z]

* CORRECT — export intercept too
scalar s_b0  = _b[_cons]
scalar s_bx  = _b[x]
scalar s_bz  = _b[z]
```

**Exception:** `oe.fe()` with one-way entity FE drops the intercept after demeaning.
In that case, do NOT export `_b[_cons]` from the `.do` file.

### 3. Use `scalar` + `clear` + `set obs` + `replace` pattern

`postfile` creates binary `.dta` that may have encoding issues. The scalar pattern is
more reliable:

```stata
* Run estimation
regress y x1 x2

* Store scalars BEFORE clear
scalar s_N   = e(N)
scalar s_b0  = _b[_cons]
scalar s_b1  = _b[x1]
scalar s_se0 = _se[_cons]
scalar s_se1 = _se[x1]

* Build output dataset
clear
set obs 5
gen str20 name  = ""
gen double value = .
replace name = "N"     in 1
replace name = "b_int" in 2
replace name = "b_x1"  in 3
replace name = "se_int" in 4
replace name = "se_x1"  in 5
replace value = s_N    in 1
replace value = s_b0   in 2
replace value = s_b1   in 3
replace value = s_se0  in 4
replace value = s_se1  in 5

save "path/to/output.dta", replace
```

### 4. Read the Stata manual before writing the command

Do NOT assume command syntax. Common mistakes:

| Assumed | Actual Stata syntax |
|---------|-------------------|
| `xtreg, fd` | Not valid in Stata 17. Use manual first-differencing: `gen dy = D.y` then `regress dy dx dz` |
| `oaxaca ... two-fold` | `oaxaca ... pooled` (two-fold) or `oaxaca ...` (three-fold is default) |
| `csdid ... gvar(binary)` | `gvar` must be treatment YEAR (0=never, 3=treat at t=3, etc.), not binary |
| `rdrobust y x` (default bw) | Different default bw selectors between packages. Specify `h()` explicitly for parity. |

### 5. Use absolute paths in `.do` files

```stata
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_ols.csv", clear
save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\ols_basic.dta", replace
```

### 6. Check the open-econs API before writing the test

Do NOT guess the API. Read the source:

- `oe.fe()` → `OLSResult` with `.coefficients`, `.std_errors`, etc.
- `oe.fe()` default `cov_type="HC2"` — Stata uses conventional SEs by default. Match with
  `cov_type="nonrobust"`.
- `oe.fe()` drops intercept after demeaning — only non-intercept coefficients returned.
- `oe.balance()` returns a DataFrame with column `Difference` (not `mean_diff`).
- `oe.event_study()` requires a `"{treatment}_event_time"` column in the data.
- `oe.staggered_did()` is OLS-based, **not** Callaway & Sant'Anna doubly-robust.
- `oe.iv()` formula: `"y ~ exog | endog ~ instruments"` (new syntax).

### 7. Match defaults exactly

If open-econs uses `cov_type="HC2"` by default but Stata uses conventional SEs, either:
- Pass `cov_type="nonrobust"` to open-econs, OR
- Document why the tolerance is relaxed

### 8. File naming convention

```
tests/stata/do/
  {estimator}_{variant}.do        # e.g., ols_basic.do, panel_fe.do
  {estimator}_{variant}.dta       # output (committed — NOT gitignored)
```

### 9. Quick checklist before committing a new `.do` file

- [ ] File is ASCII/UTF-8 without BOM
- [ ] Uses absolute paths for import and save
- [ ] Exports ALL coefficients including intercept (unless oe drops it)
- [ ] Command syntax verified against `help {command}` in Stata
- [ ] Scalars saved BEFORE `clear`
- [ ] Output `.dta` is committed alongside the `.do`
- [ ] Both `.do` and `.dta` are committed together

### 10. Tolerance guidelines

Cross-platform float precision between Python (statsmodels) and Stata tops out
around 1e-7 — these are genuine floating-point noise from different BLAS/LAPACK
routines, not algorithmic discrepancies.  Use these floors:

| Estimator family | Tolerance | Why |
|-----------------|-----------|-----|
| OLS, Logit, Probit, IV, DiD, FE, FD | `rtol=1e-7` | Same algorithm, cross-platform float noise floor |
| HAC | `rtol=1e-2` | Different kernel implementations |
| Abond, RDD, Staggered DiD, Oaxaca | Absolute thresholds | Fundamentally different algorithms |
| Logit/Probit margins | `rtol=0.8` | MEM (oe) vs AME (Stata) — different by definition |
| Panel RE, Pooled, Hausman | `rtol=1e-7` target | Investigate root cause if mismatch, don't relax |

**Rule:** If a test needs relaxed tolerance, document WHY in the test comment.
Never relax tolerance as a shortcut to make a failing test pass.
