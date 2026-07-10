*! logit_margins.do — Logit marginal effects (AME)
* Computes Average Marginal Effects (AME) to match Stata's default margins
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_logit.csv", clear

* Estimate logit
logit y x1 x2

* Compute marginal effects (AME - average over all observations)
margins, dydx(x1 x2)

* Extract marginal effect coefficients using r(b)
scalar s_me1 = r(b)[1,1]
scalar s_me2 = r(b)[1,2]

clear
set obs 2
gen str20 name  = ""
gen double value = .
replace name = "me_x1" in 1
replace name = "me_x2" in 2
replace value = s_me1  in 1
replace value = s_me2  in 2

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\logit_margins.dta", replace
