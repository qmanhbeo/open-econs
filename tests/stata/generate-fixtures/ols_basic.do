*! ols_basic.do — OLS with classical SEs
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_ols.csv", clear
regress y x1 x2

* Store results
scalar s_N     = e(N)
scalar s_df_r  = e(df_r)
scalar s_df_m  = e(df_m)
scalar s_r2    = e(r2)
scalar s_r2_a  = e(r2_a)
scalar s_F     = e(F)
scalar s_p     = e(p)
scalar s_b0    = _b[_cons]
scalar s_b1    = _b[x1]
scalar s_b2    = _b[x2]
scalar s_se0   = _se[_cons]
scalar s_se1   = _se[x1]
scalar s_se2   = _se[x2]

* Build output dataset
clear
set obs 13
gen str20 name  = ""
gen double value = .

replace name = "N"      in 1
replace name = "df_m"   in 2
replace name = "df_r"   in 3
replace name = "r2"     in 4
replace name = "r2_a"   in 5
replace name = "F"      in 6
replace name = "pval"   in 7
replace name = "b_int"  in 8
replace name = "b_x1"   in 9
replace name = "b_x2"   in 10
replace name = "se_int" in 11
replace name = "se_x1"  in 12
replace name = "se_x2"  in 13

replace value = s_N    in 1
replace value = s_df_m in 2
replace value = s_df_r in 3
replace value = s_r2   in 4
replace value = s_r2_a in 5
replace value = s_F    in 6
replace value = s_p    in 7
replace value = s_b0   in 8
replace value = s_b1   in 9
replace value = s_b2   in 10
replace value = s_se0  in 11
replace value = s_se1  in 12
replace value = s_se2  in 13

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\ols_basic.dta", replace
