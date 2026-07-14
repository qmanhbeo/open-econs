*! panel_hausman.do — Hausman test (FE vs RE)
*! Stores the DISPLAYED chi2, not e(chi2) which is a ghost variable.
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv", clear
xtset entity time

quietly xtreg y x z, fe
estimates store fe_model
matrix b_fe_full = e(b)
matrix V_fe_full = e(V)

quietly xtreg y x z, re
estimates store re_model
matrix b_re_full = e(b)
matrix V_re_full = e(V)

* Run hausman for display purposes
hausman fe_model re_model

* Compute the DISPLAYED chi2 manually from the quadratic form
* (e(chi2) is unreliable — it stores a different statistic)
matrix b_fe = b_fe_full[1, 1..2]
matrix b_re = b_re_full[1, 1..2]
matrix V_fe = V_fe_full[1..2, 1..2]
matrix V_re = V_re_full[1..2, 1..2]
matrix diff = b_fe - b_re
matrix V_diff = V_fe - V_re
matrix H = diff * invsym(V_diff) * diff'
scalar s_chi2 = H[1,1]

* Compute p-value from chi2(2) distribution
scalar s_p = 1 - chi2(2, s_chi2)

clear
set obs 2
gen str20 name  = ""
gen double value = .
replace name = "chi2" in 1
replace name = "p"    in 2
replace value = s_chi2 in 1
replace value = s_p    in 2

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\panel_hausman.dta", replace
