*! iv_basic.do â€” IV / 2SLS
clear all
set more off
set type double
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_iv.csv", clear
ivregress 2sls y x2 (x = z)

scalar s_N    = e(N)
scalar s_b0   = _b[_cons]
scalar s_bx2  = _b[x2]
scalar s_bx   = _b[x]
scalar s_se0  = _se[_cons]
scalar s_sex2 = _se[x2]
scalar s_sex  = _se[x]

clear
set obs 7
gen str20 name  = ""
gen double value = .
replace name = "N"     in 1
replace name = "b_int" in 2
replace name = "b_x2"  in 3
replace name = "b_x"   in 4
replace name = "se_int" in 5
replace name = "se_x2" in 6
replace name = "se_x"  in 7
replace value = s_N    in 1
replace value = s_b0   in 2
replace value = s_bx2  in 3
replace value = s_bx   in 4
replace value = s_se0  in 5
replace value = s_sex2 in 6
replace value = s_sex  in 7

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\iv_basic.dta", replace
