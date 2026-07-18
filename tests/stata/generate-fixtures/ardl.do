*! ARDL / UECM PSS bounds fixture generation for open-econs parity tests
*! Canonical example: denmark data, LRM ~ LRY + IBO + IDE, lags(3 1 3 2), PSS case 3
*! Source-verified against c:\ado\plus\a\ardl.ado (v1.0.6):
*!   - e(F_pss): Wald F on LR level vars + LR deterministics (PSS F-stat)
*!   - e(t_pss): t-stat on L.depvar
*!   - e(F_critval) / e(t_critval): 1 x 8, cols =
*!       1:10%lo 2:10%hi 3:5%lo 4:5%hi 5:2.5%lo 6:2.5%hi 7:1%lo 8:1%hi
*!     from ardlbounds, table nosurfreg (PSS 2001 published table, nsource=pssmith)
*!   - e(b) has equations ADJ / LR / SR. EC term = coef on L.depvar (ADJ eq).
*!     LR coefficients (long-run multipliers) are the LR-equation cols for the
*!     level regressors L.lry L.ibo L.ide (cols 2,3,4 of e(b)). Coefficients are
*!     equation-qualified, so they are extracted by matrix position, not _b[].

clear all
set more off

* ROOT CAUSE / FOOTGUN (rule 18): `import delimited` reads numeric columns as
* SINGLE-precision (float) by default. That truncation of the input data shifts
* the OLS coefficients (R^2=0.988, near-collinear lags) at ~3e-6 and the PSS F at
* ~4e-5, producing a spurious Stata-vs-(R/statsmodels) divergence. R read.csv and
* pandas read.csv both default to double, so ONLY the Stata side was affected.
* `set type double` makes import read double precision; with it, e(F_pss)/e(t_pss)
* /EC/LR match OE (= R = statsmodels) to <1e-6. See methodology/timeseries/ardl.md.
set type double

import delimited "tests/r/fixtures/inputs/ardl_input.csv", clear
gen t = _n
tsset t

* EC representation; case 3 (unrestricted constant) by default
ardl lrm lry ibo ide, lags(3 1 3 2) ec

* --- PSS bounds statistics ---
scalar Fpss = e(F_pss)
scalar tpss = e(t_pss)
matrix Fcv = e(F_critval)
matrix tcv = e(t_critval)

* --- EC term and long-run multipliers (extracted by matrix position) ---
matrix b = e(b)
scalar ec_term = b[1, 1]          // ADJ: L.lrm  (speed of adjustment)
scalar lr_LRY   = b[1, 2]          // LR: L.lry
scalar lr_IBO   = b[1, 3]          // LR: L.ibo
scalar lr_IDE   = b[1, 4]          // LR: L.ide

* --- Build name/value dataset ---
clear
set obs 50
gen str32 name = ""
gen double value = .

local i = 1
replace name = "f_stat" in `i'
replace value = Fpss in `i'

local i = `i' + 1
replace name = "t_stat" in `i'
replace value = tpss in `i'

local i = `i' + 1
replace name = "ec_term" in `i'
replace value = ec_term in `i'

local i = `i' + 1
replace name = "lr_LRY" in `i'
replace value = lr_LRY in `i'

local i = `i' + 1
replace name = "lr_IBO" in `i'
replace value = lr_IBO in `i'

local i = `i' + 1
replace name = "lr_IDE" in `i'
replace value = lr_IDE in `i'

* F critical values: cols 1..8 = 10%lo,10%hi,5%lo,5%hi,2.5%lo,2.5%hi,1%lo,1%hi
local i = `i' + 1
replace name = "f_cv_lower_10" in `i'
replace value = Fcv[1, 1] in `i'
local i = `i' + 1
replace name = "f_cv_upper_10" in `i'
replace value = Fcv[1, 2] in `i'
local i = `i' + 1
replace name = "f_cv_lower_5" in `i'
replace value = Fcv[1, 3] in `i'
local i = `i' + 1
replace name = "f_cv_upper_5" in `i'
replace value = Fcv[1, 4] in `i'
local i = `i' + 1
replace name = "f_cv_lower_25" in `i'
replace value = Fcv[1, 5] in `i'
local i = `i' + 1
replace name = "f_cv_upper_25" in `i'
replace value = Fcv[1, 6] in `i'
local i = `i' + 1
replace name = "f_cv_lower_1" in `i'
replace value = Fcv[1, 7] in `i'
local i = `i' + 1
replace name = "f_cv_upper_1" in `i'
replace value = Fcv[1, 8] in `i'

* t critical values: cols 1..8 = 10%lo,10%hi,5%lo,5%hi,2.5%lo,2.5%hi,1%lo,1%hi
local i = `i' + 1
replace name = "t_cv_lower_10" in `i'
replace value = tcv[1, 1] in `i'
local i = `i' + 1
replace name = "t_cv_upper_10" in `i'
replace value = tcv[1, 2] in `i'
local i = `i' + 1
replace name = "t_cv_lower_5" in `i'
replace value = tcv[1, 3] in `i'
local i = `i' + 1
replace name = "t_cv_upper_5" in `i'
replace value = tcv[1, 4] in `i'
local i = `i' + 1
replace name = "t_cv_lower_25" in `i'
replace value = tcv[1, 5] in `i'
local i = `i' + 1
replace name = "t_cv_upper_25" in `i'
replace value = tcv[1, 6] in `i'
local i = `i' + 1
replace name = "t_cv_lower_1" in `i'
replace value = tcv[1, 7] in `i'
local i = `i' + 1
replace name = "t_cv_upper_1" in `i'
replace value = tcv[1, 8] in `i'

drop if name == ""
save "tests/stata/fixtures/expected/ardl.dta", replace
