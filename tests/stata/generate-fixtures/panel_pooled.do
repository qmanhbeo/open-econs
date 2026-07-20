*! panel_pooled.do â€” Pooled OLS (regress)
clear all
set more off
set type double
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear
regress y x z

scalar s_N   = e(N)
scalar s_b0  = _b[_cons]
scalar s_bx  = _b[x]
scalar s_bz  = _b[z]
scalar s_se0 = _se[_cons]
scalar s_sex = _se[x]
scalar s_sez = _se[z]

clear
set obs 7
gen str20 name  = ""
gen double value = .
replace name = "N"     in 1
replace name = "b_int" in 2
replace name = "b_x"   in 3
replace name = "b_z"   in 4
replace name = "se_int" in 5
replace name = "se_x"  in 6
replace name = "se_z"  in 7
replace value = s_N    in 1
replace value = s_b0   in 2
replace value = s_bx   in 3
replace value = s_bz   in 4
replace value = s_se0  in 5
replace value = s_sex  in 6
replace value = s_sez  in 7

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\panel_pooled.dta", replace
