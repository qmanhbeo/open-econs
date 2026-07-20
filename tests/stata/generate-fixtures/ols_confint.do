*! ols_confint.do â€” OLS confidence intervals
clear all
set more off
set type double
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_ols.csv", clear
regress y x1 x2

scalar tval = invt(e(df_r), 0.975)
scalar s_b0  = _b[_cons]
scalar s_b1  = _b[x1]
scalar s_b2  = _b[x2]
scalar s_se0 = _se[_cons]
scalar s_se1 = _se[x1]
scalar s_se2 = _se[x2]

clear
set obs 9
gen str20 name  = ""
gen double value = .
replace name = "b_int"     in 1
replace name = "b_int_ll"  in 2
replace name = "b_int_ul"  in 3
replace name = "b_x1"      in 4
replace name = "b_x1_ll"   in 5
replace name = "b_x1_ul"   in 6
replace name = "b_x2"      in 7
replace name = "b_x2_ll"   in 8
replace name = "b_x2_ul"   in 9

replace value = s_b0                  in 1
replace value = s_b0 - tval * s_se0   in 2
replace value = s_b0 + tval * s_se0   in 3
replace value = s_b1                  in 4
replace value = s_b1 - tval * s_se1   in 5
replace value = s_b1 + tval * s_se1   in 6
replace value = s_b2                  in 7
replace value = s_b2 - tval * s_se2   in 8
replace value = s_b2 + tval * s_se2   in 9

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\ols_confint.dta", replace
