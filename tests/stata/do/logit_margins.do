*! logit_margins.do — Logit marginal effects
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_logit.csv", clear
logit y x1 x2
margins, dydx(x1 x2)

matrix m = e(b)'
scalar s_me1 = m[1,1]
scalar s_me2 = m[2,1]

clear
set obs 2
gen str20 name  = ""
gen double value = .
replace name = "me_x1" in 1
replace name = "me_x2" in 2
replace value = s_me1  in 1
replace value = s_me2  in 2

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\logit_margins.dta", replace
