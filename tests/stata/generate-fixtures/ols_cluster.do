*! ols_cluster.do â€” OLS with single-way cluster SEs
clear all
set more off
set type double
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_ols.csv", clear
regress y x1 x2, cluster(province)

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

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\ols_cluster.dta", replace
