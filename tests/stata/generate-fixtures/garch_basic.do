clear
import delimited using "tests/r/fixtures/inputs/garch_input.csv", clear
gen t = _n
tsset t
arch y, arch(1) garch(1) nolog
clear
set obs 5
gen str20 name = ""
gen double value = .
replace name = "mu"     in 1
replace value = el(e(b),1,1) in 1
replace name = "omega"  in 2
replace value = el(e(b),1,4) in 2
replace name = "alpha"  in 3
replace value = el(e(b),1,2) in 3
replace name = "beta"   in 4
replace value = el(e(b),1,3) in 4
replace name = "ll"     in 5
replace value = e(ll)         in 5
save "tests/stata/fixtures/expected/garch_basic.dta", replace
