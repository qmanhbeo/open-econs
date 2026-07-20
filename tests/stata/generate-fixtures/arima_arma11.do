clear
set type double
import delimited using "tests/r/fixtures/inputs/arma_input.csv", clear
gen t = _n
tsset t
arima y, ar(1) ma(1)
clear
set obs 5
gen str20 name = ""
gen double value = .
replace name = "cons" in 1
replace value = _b[_cons]     in 1
replace name = "ar1"  in 2
replace value = _b[ARMA:L.ar] in 2
replace name = "ma1"  in 3
replace value = _b[ARMA:L.ma] in 3
replace name = "ll"   in 4
replace value = e(ll)         in 4
replace name = "nobs" in 5
replace value = e(N)          in 5
save "tests/stata/fixtures/expected/arima_arma11.dta", replace
