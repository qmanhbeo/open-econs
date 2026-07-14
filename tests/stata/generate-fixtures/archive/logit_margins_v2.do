*! logit_margins.do — Logit marginal effects (AME)
* Version 2: explicitly store margins results
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_logit.csv", clear

* Estimate logit
logit y x1 x2

* Compute marginal effects (AME - average over all observations)
margins, dydx(x1 x2)

* After margins, _b should contain the marginal effects
* Let's verify by storing in a matrix first
matrix me = r(table)'
matrix list me
matrix list r(b)

display "x1 from _b: " _b[x1]
display "x2 from _b: " _b[x2]
display "x1 from r(b): " r(b)[1,1]
display "x2 from r(b): " r(b)[1,2]

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

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\logit_margins.dta", replace