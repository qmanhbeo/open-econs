*! balance_basic.do — Covariate balance (Welch t-tests, Treated - Control)
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_ols.csv", clear
gen treat = (province == "north")

* Welch (unequal-variance) t-test matching scipy.stats.ttest_ind(equal_var=False)
* r(mu_2) = treat==1 (treated) , r(mu_1) = treat==0 (control)
* Signed to give "treated - control", matching open-econs convention
quietly ttest x1, by(treat) unequal
scalar s_d1 = r(mu_2) - r(mu_1)
scalar s_t1 = -r(t)
scalar s_df1 = r(df_t)
scalar s_p1 = r(p)

quietly ttest x2, by(treat) unequal
scalar s_d2 = r(mu_2) - r(mu_1)
scalar s_t2 = -r(t)
scalar s_df2 = r(df_t)
scalar s_p2 = r(p)

clear
set obs 10
gen str20 name  = ""
gen double value = .
replace name = "diff_x1" in 1
replace name = "t_x1"    in 2
replace name = "df_x1"   in 3
replace name = "p_x1"    in 4
replace name = "diff_x2" in 5
replace name = "t_x2"    in 6
replace name = "df_x2"   in 7
replace name = "p_x2"    in 8
replace value = s_d1  in 1
replace value = s_t1  in 2
replace value = s_df1 in 3
replace value = s_p1  in 4
replace value = s_d2  in 5
replace value = s_t2  in 6
replace value = s_df2 in 7
replace value = s_p2  in 8

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\balance_basic.dta", replace
