clear
set type double
import delimited using "tests/r/fixtures/inputs/ur_input.csv", clear
gen t = _n
tsset t
pperron y
clear
set obs 4
gen str20 name = ""
gen double value = .
replace name = "stat"   in 1
replace value = r(Zt)   in 1
replace name = "pvalue" in 2
replace value = r(pval) in 2
replace name = "lags"   in 3
replace value = r(lags) in 3
replace name = "nobs"   in 4
replace value = r(N)    in 4
save "tests/stata/fixtures/expected/ur_pp_c.dta", replace
