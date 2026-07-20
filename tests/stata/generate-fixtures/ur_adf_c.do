clear
set type double
import delimited using "tests/r/fixtures/inputs/ur_input.csv", clear
gen t = _n
tsset t
dfuller y
clear
set obs 7
gen str20 name = ""
gen double value = .
replace name = "stat"   in 1
replace value = r(Zt)   in 1
replace name = "pvalue" in 2
replace value = r(p)    in 2
replace name = "lags"   in 3
replace value = r(lags) in 3
replace name = "nobs"   in 4
replace value = r(N)    in 4
replace name = "cv_1"   in 5
replace value = r(cv_1) in 5
replace name = "cv_5"   in 6
replace value = r(cv_5) in 6
replace name = "cv_10"  in 7
replace value = r(cv_10) in 7
save "tests/stata/fixtures/expected/ur_adf_c.dta", replace
