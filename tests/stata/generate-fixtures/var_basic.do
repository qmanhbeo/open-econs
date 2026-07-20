*! VAR/VECM fixture generation for open-econs parity tests
*! Uses var_input.csv (200 obs, 2 variables: y1, y2)
*!
*! Johansen cases (Osterwald-Lenum 1992):
*!   Case 1: trend(none)       â€” no deterministic term
*!   Case 2: trend(rconstant)  â€” restricted constant (in CE only)
*!   Case 3: trend(constant)   â€” unrestricted constant (default)
*!   Case 4: trend(rtrend)     â€” restricted trend (in CE only)
*!   Case 5: trend(trend)      â€” unrestricted trend

clear all
set more off

* Import data
set type double
import delimited "tests/r/fixtures/inputs/var_input.csv", clear

* Create time index for tsset
gen t = _n
tsset t

* â”€â”€ 1. VAR lag order selection (varsoc) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
var y1 y2, lags(1/5)
varsoc, max(5)

* â”€â”€ 2. VAR estimation at lag 2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
var y1 y2, lags(1/2)
estat ic

* Store VAR results
* Note: Stata VAR stores e(sbic) (per-obs BIC); estat ic BIC = N * e(sbic)
scalar ll_var = e(ll)
scalar aic_var = e(aic)
scalar bic_var = e(N) * e(sbic)
scalar hqic_var = e(hqic)

* â”€â”€ 3. Johansen cointegration â€” all 5 cases â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
* vecrank uses trend() option: none/rconstant/constant/rtrend/trend
* For 2 variables: e(trace) and e(max) are 1x2 matrices [r<=0, r<=1]
* CVs from _vecgetcv are 11x5 matrices [K-r x trend_type]

* Case 1: no deterministic term (trend col = 1)
vecrank y1 y2, lags(2) trend(none)
matrix tr1 = e(trace)
matrix me1 = e(max)
_vecgetcv trace
matrix cv_tr1 = r(cv95)
_vecgetcv max
matrix cv_me1 = r(cv95)

* Case 2: restricted constant (trend col = 2)
vecrank y1 y2, lags(2) trend(rconstant)
matrix tr2 = e(trace)
matrix me2 = e(max)
_vecgetcv trace
matrix cv_tr2 = r(cv95)
_vecgetcv max
matrix cv_me2 = r(cv95)

* Case 3: unrestricted constant (trend col = 3)
vecrank y1 y2, lags(2) trend(constant)
matrix tr3 = e(trace)
matrix me3 = e(max)
_vecgetcv trace
matrix cv_tr3 = r(cv95)
_vecgetcv max
matrix cv_me3 = r(cv95)

* Case 4: restricted trend (trend col = 4)
vecrank y1 y2, lags(2) trend(rtrend)
matrix tr4 = e(trace)
matrix me4 = e(max)
_vecgetcv trace
matrix cv_tr4 = r(cv95)
_vecgetcv max
matrix cv_me4 = r(cv95)

* Case 5: unrestricted trend (trend col = 5)
vecrank y1 y2, lags(2) trend(trend)
matrix tr5 = e(trace)
matrix me5 = e(max)
_vecgetcv trace
matrix cv_tr5 = r(cv95)
_vecgetcv max
matrix cv_me5 = r(cv95)

* â”€â”€ 4. Granger causality (vargranger) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
* Re-estimate VAR for vargranger
var y1 y2, lags(1/2)

* Default: chi-squared (Wald)
vargranger

* â”€â”€ 5. Save results as .dta â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
* Structure: name-value pairs for all quantities
* For 2 variables, vecrank produces 2 test statistics per test type per case

clear
set obs 100
gen str32 name = ""
gen double value = .

* VAR IC
local i = 1
replace name = "ll_var" in `i'
replace value = ll_var in `i'
local i = `i' + 1
replace name = "aic_var" in `i'
replace value = aic_var in `i'
local i = `i' + 1
replace name = "bic_var" in `i'
replace value = bic_var in `i'
local i = `i' + 1
replace name = "hqic_var" in `i'
replace value = hqic_var in `i'

* â”€â”€ Johansen trace statistics (e(trace) is 1 x k, columns = r<=0, r<=1) â”€â”€

* Case 1: trace
local i = `i' + 1
replace name = "trace_case1_r1" in `i'
replace value = tr1[1, 1] in `i'
local i = `i' + 1
replace name = "trace_case1_r2" in `i'
replace value = tr1[1, 2] in `i'

* Case 2: trace
local i = `i' + 1
replace name = "trace_case2_r1" in `i'
replace value = tr2[1, 1] in `i'
local i = `i' + 1
replace name = "trace_case2_r2" in `i'
replace value = tr2[1, 2] in `i'

* Case 3: trace
local i = `i' + 1
replace name = "trace_case3_r1" in `i'
replace value = tr3[1, 1] in `i'
local i = `i' + 1
replace name = "trace_case3_r2" in `i'
replace value = tr3[1, 2] in `i'

* Case 4: trace
local i = `i' + 1
replace name = "trace_case4_r1" in `i'
replace value = tr4[1, 1] in `i'
local i = `i' + 1
replace name = "trace_case4_r2" in `i'
replace value = tr4[1, 2] in `i'

* Case 5: trace
local i = `i' + 1
replace name = "trace_case5_r1" in `i'
replace value = tr5[1, 1] in `i'
local i = `i' + 1
replace name = "trace_case5_r2" in `i'
replace value = tr5[1, 2] in `i'

* â”€â”€ Johansen max-eigenvalue statistics (e(max) is 1 x k) â”€â”€â”€â”€â”€â”€â”€

* Case 1: maxeig
local i = `i' + 1
replace name = "maxeig_case1_r1" in `i'
replace value = me1[1, 1] in `i'
local i = `i' + 1
replace name = "maxeig_case1_r2" in `i'
replace value = me1[1, 2] in `i'

* Case 2: maxeig
local i = `i' + 1
replace name = "maxeig_case2_r1" in `i'
replace value = me2[1, 1] in `i'
local i = `i' + 1
replace name = "maxeig_case2_r2" in `i'
replace value = me2[1, 2] in `i'

* Case 3: maxeig
local i = `i' + 1
replace name = "maxeig_case3_r1" in `i'
replace value = me3[1, 1] in `i'
local i = `i' + 1
replace name = "maxeig_case3_r2" in `i'
replace value = me3[1, 2] in `i'

* Case 4: maxeig
local i = `i' + 1
replace name = "maxeig_case4_r1" in `i'
replace value = me4[1, 1] in `i'
local i = `i' + 1
replace name = "maxeig_case4_r2" in `i'
replace value = me4[1, 2] in `i'

* Case 5: maxeig
local i = `i' + 1
replace name = "maxeig_case5_r1" in `i'
replace value = me5[1, 1] in `i'
local i = `i' + 1
replace name = "maxeig_case5_r2" in `i'
replace value = me5[1, 2] in `i'

* â”€â”€ Johansen 5% CVs from _vecgetcv â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
* cv95 is 11x5: rows = K-r (1..11), cols = trend type (1..5)
* For 2 vars: K=2, so K-r=2 at r=0, K-r=1 at r=1

* Case 1 (trend col=1): cv_trace5
local i = `i' + 1
replace name = "cv_trace5_case1_r1" in `i'
replace value = cv_tr1[2, 1] in `i'
local i = `i' + 1
replace name = "cv_trace5_case1_r2" in `i'
replace value = cv_tr1[1, 1] in `i'

* Case 2 (trend col=2): cv_trace5
local i = `i' + 1
replace name = "cv_trace5_case2_r1" in `i'
replace value = cv_tr2[2, 2] in `i'
local i = `i' + 1
replace name = "cv_trace5_case2_r2" in `i'
replace value = cv_tr2[1, 2] in `i'

* Case 3 (trend col=3): cv_trace5
local i = `i' + 1
replace name = "cv_trace5_case3_r1" in `i'
replace value = cv_tr3[2, 3] in `i'
local i = `i' + 1
replace name = "cv_trace5_case3_r2" in `i'
replace value = cv_tr3[1, 3] in `i'

* Case 4 (trend col=4): cv_trace5
local i = `i' + 1
replace name = "cv_trace5_case4_r1" in `i'
replace value = cv_tr4[2, 4] in `i'
local i = `i' + 1
replace name = "cv_trace5_case4_r2" in `i'
replace value = cv_tr4[1, 4] in `i'

* Case 5 (trend col=5): cv_trace5
local i = `i' + 1
replace name = "cv_trace5_case5_r1" in `i'
replace value = cv_tr5[2, 5] in `i'
local i = `i' + 1
replace name = "cv_trace5_case5_r2" in `i'
replace value = cv_tr5[1, 5] in `i'

* â”€â”€ Johansen 5% CVs â€” max-eigenvalue â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

* Case 1 (trend col=1): cv_maxeig5
local i = `i' + 1
replace name = "cv_maxeig5_case1_r1" in `i'
replace value = cv_me1[2, 1] in `i'
local i = `i' + 1
replace name = "cv_maxeig5_case1_r2" in `i'
replace value = cv_me1[1, 1] in `i'

* Case 2 (trend col=2): cv_maxeig5
local i = `i' + 1
replace name = "cv_maxeig5_case2_r1" in `i'
replace value = cv_me2[2, 2] in `i'
local i = `i' + 1
replace name = "cv_maxeig5_case2_r2" in `i'
replace value = cv_me2[1, 2] in `i'

* Case 3 (trend col=3): cv_maxeig5
local i = `i' + 1
replace name = "cv_maxeig5_case3_r1" in `i'
replace value = cv_me3[2, 3] in `i'
local i = `i' + 1
replace name = "cv_maxeig5_case3_r2" in `i'
replace value = cv_me3[1, 3] in `i'

* Case 4 (trend col=4): cv_maxeig5
local i = `i' + 1
replace name = "cv_maxeig5_case4_r1" in `i'
replace value = cv_me4[2, 4] in `i'
local i = `i' + 1
replace name = "cv_maxeig5_case4_r2" in `i'
replace value = cv_me4[1, 4] in `i'

* Case 5 (trend col=5): cv_maxeig5
local i = `i' + 1
replace name = "cv_maxeig5_case5_r1" in `i'
replace value = cv_me5[2, 5] in `i'
local i = `i' + 1
replace name = "cv_maxeig5_case5_r2" in `i'
replace value = cv_me5[1, 5] in `i'

* Drop empty rows
drop if name == ""

save "tests/stata/fixtures/expected/var_basic.dta", replace
