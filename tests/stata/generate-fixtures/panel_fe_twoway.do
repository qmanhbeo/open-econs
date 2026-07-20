*! panel_fe_twoway.do â€” Two-way FE: entity + time dummies
clear all
set more off
set type double
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear
xtset entity time
xtreg y x z i.time, fe

scalar s_N      = e(N)
scalar s_N_g    = e(N_g)
scalar s_r2_w   = e(r2_w)
scalar s_df_r   = e(df_r)
scalar s_bx     = _b[x]
scalar s_bz     = _b[z]
scalar s_sex    = _se[x]
scalar s_sez    = _se[z]

clear
set obs 8
gen str20 name  = ""
gen double value = .
replace name = "N"    in 1
replace name = "N_g"  in 2
replace name = "r2_w" in 3
replace name = "df_r" in 4
replace name = "b_x"  in 5
replace name = "b_z"  in 6
replace name = "se_x" in 7
replace name = "se_z" in 8
replace value = s_N    in 1
replace value = s_N_g  in 2
replace value = s_r2_w in 3
replace value = s_df_r in 4
replace value = s_bx   in 5
replace value = s_bz   in 6
replace value = s_sex  in 7
replace value = s_sez  in 8

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\panel_fe_twoway.dta", replace
