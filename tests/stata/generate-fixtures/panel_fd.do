*! panel_fd.do — First difference estimator via manual differencing
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear
xtset entity time
gen dy = D.y
gen dx = D.x
gen dz = D.z
regress dy dx dz, noconstant

scalar s_N   = e(N)
scalar s_bx  = _b[dx]
scalar s_bz  = _b[dz]
scalar s_sex = _se[dx]
scalar s_sez = _se[dz]

clear
set obs 5
gen str20 name  = ""
gen double value = .
replace name = "N"    in 1
replace name = "b_x"  in 2
replace name = "b_z"  in 3
replace name = "se_x" in 4
replace name = "se_z" in 5
replace value = s_N   in 1
replace value = s_bx  in 2
replace value = s_bz  in 3
replace value = s_sex in 4
replace value = s_sez in 5

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\panel_fd.dta", replace
