*! logit_basic.do — Binary logit
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_logit.csv", clear
logit y x1 x2

scalar s_N    = e(N)
scalar s_bx1  = _b[x1]
scalar s_bx2  = _b[x2]
scalar s_sex1 = _se[x1]
scalar s_sex2 = _se[x2]

clear
set obs 5
gen str20 name  = ""
gen double value = .
replace name = "N"     in 1
replace name = "b_x1"  in 2
replace name = "b_x2"  in 3
replace name = "se_x1" in 4
replace name = "se_x2" in 5
replace value = s_N    in 1
replace value = s_bx1  in 2
replace value = s_bx2  in 3
replace value = s_sex1 in 4
replace value = s_sex2 in 5

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\logit_basic.dta", replace
