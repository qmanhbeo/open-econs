*! balance_basic.do — Covariate balance (t-tests)
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_ols.csv", clear
gen treat = (province == "north")

quietly ttest x1, by(treat)
scalar s_d1 = r(mu_1) - r(mu_2)
scalar s_t1 = r(t)

quietly ttest x2, by(treat)
scalar s_d2 = r(mu_1) - r(mu_2)
scalar s_t2 = r(t)

clear
set obs 4
gen str20 name  = ""
gen double value = .
replace name = "diff_x1" in 1
replace name = "t_x1"    in 2
replace name = "diff_x2" in 3
replace name = "t_x2"    in 4
replace value = s_d1 in 1
replace value = s_t1 in 2
replace value = s_d2 in 3
replace value = s_t2 in 4

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\balance_basic.dta", replace
