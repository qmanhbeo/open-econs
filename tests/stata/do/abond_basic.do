*! abond_basic.do — Arellano-Bond GMM (SSC: xtabond2)
clear all
set more off
capture ssc install xtabond2
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv", clear
xtset entity time
xtabond2 y L.y x z, gmm(L.y, lag(1 2)) iv(x z) twostep robust small

scalar s_N    = e(N)
scalar s_bly  = _b[L.y]
scalar s_bx   = _b[x]
scalar s_bz   = _b[z]
scalar s_sely = _se[L.y]
scalar s_sex  = _se[x]
scalar s_sez  = _se[z]

clear
set obs 7
gen str20 name  = ""
gen double value = .
replace name = "N"     in 1
replace name = "b_Ly"  in 2
replace name = "b_x"   in 3
replace name = "b_z"   in 4
replace name = "se_Ly" in 5
replace name = "se_x"  in 6
replace name = "se_z"  in 7
replace value = s_N    in 1
replace value = s_bly  in 2
replace value = s_bx   in 3
replace value = s_bz   in 4
replace value = s_sely in 5
replace value = s_sex  in 6
replace value = s_sez  in 7

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\abond_basic.dta", replace
