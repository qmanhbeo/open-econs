*! panel_fe_multiway_cluster.do — FE with multiway cluster SEs (reghdfe)
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv", clear

* Two-way cluster on entity and time, absorb both as FE
reghdfe y x z, absorb(entity time) vce(cluster entity time)

scalar s_N      = e(N)
scalar s_N_g    = e(N_g)
scalar s_r2_a   = e(r2_a)
scalar s_df_r   = e(df_r)
scalar s_bx     = _b[x]
scalar s_bz     = _b[z]
scalar s_sex    = _se[x]
scalar s_sez    = _se[z]
scalar s_fstat  = e(F)

clear
set obs 10
gen str20 name  = ""
gen double value = .
replace name = "N"      in 1
replace name = "N_g"    in 2
replace name = "r2_a"   in 3
replace name = "df_r"   in 4
replace name = "b_x"    in 5
replace name = "b_z"    in 6
replace name = "se_x"   in 7
replace name = "se_z"   in 8
replace name = "fstat"  in 9
replace name = "se_int" in 10
replace value = s_N      in 1
replace value = s_N_g    in 2
replace value = s_r2_a   in 3
replace value = s_df_r   in 4
replace value = s_bx     in 5
replace value = s_bz     in 6
replace value = s_sex    in 7
replace value = s_sez    in 8
replace value = s_fstat  in 9
replace value = 0        in 10

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\panel_fe_multiway_cluster.dta", replace
