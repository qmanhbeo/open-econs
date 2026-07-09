*! event_study.do — Event study (relative-time dummies)
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_did.csv", clear
gen treat_post = treat * post
regress y treat post treat_post

scalar s_b0  = _b[_cons]
scalar s_se0 = _se[_cons]

clear
set obs 2
gen str20 name  = ""
gen double value = .
replace name = "b_int"  in 1
replace name = "se_int" in 2
replace value = s_b0    in 1
replace value = s_se0   in 2

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\event_study.dta", replace
