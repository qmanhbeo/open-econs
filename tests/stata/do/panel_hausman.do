*! panel_hausman.do — Hausman test (FE vs RE)
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv", clear
xtset entity time

quietly xtreg y x z, fe
estimates store fe_model
quietly xtreg y x z, re
estimates store re_model
hausman fe_model re_model

scalar s_chi2 = e(chi2)
scalar s_p    = e(p)

clear
set obs 2
gen str20 name  = ""
gen double value = .
replace name = "chi2" in 1
replace name = "p"    in 2
replace value = s_chi2 in 1
replace value = s_p    in 2

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\panel_hausman.dta", replace
