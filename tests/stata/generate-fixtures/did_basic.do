*! did_basic.do â€” Two-period DiD
clear all
set more off
set type double
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_did.csv", clear
gen treat_post = treat * post
regress y treat post treat_post

scalar s_N     = e(N)
scalar s_b_int = _b[_cons]
scalar s_b_did = _b[treat_post]
scalar s_se_did = _se[treat_post]

clear
set obs 4
gen str20 name  = ""
gen double value = .
replace name = "N"           in 1
replace name = "b_int"       in 2
replace name = "b_treatXpost" in 3
replace name = "se_treatXpost" in 4
replace value = s_N     in 1
replace value = s_b_int in 2
replace value = s_b_did in 3
replace value = s_se_did in 4

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\did_basic.dta", replace
