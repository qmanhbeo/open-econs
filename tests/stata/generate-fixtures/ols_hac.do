*! ols_hac.do — OLS with Newey-West HAC SEs (lag 2)
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_ols.csv", clear
gen t = _n
tsset t
newey y x1 x2, lag(2)

scalar s_se0 = _se[_cons]
scalar s_se1 = _se[x1]
scalar s_se2 = _se[x2]

clear
set obs 3
gen str20 name  = ""
gen double value = .
replace name = "se_int" in 1
replace name = "se_x1"  in 2
replace name = "se_x2"  in 3
replace value = s_se0 in 1
replace value = s_se1 in 2
replace value = s_se2 in 3

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\ols_hac.dta", replace
