*! rdd_fuzzy.do — Fuzzy RDD (SSC: rdrobust)
clear all
set more off
capture ssc install rdrobust
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_rdd.csv", clear
rdrobust y_fuzzy x, c(0) fuzzy(treat)

scalar s_bw   = e(h_l)
scalar s_coef = e(tau_cl)
scalar s_se   = e(se_tau_cl)

clear
set obs 3
gen str20 name  = ""
gen double value = .
replace name = "bw"   in 1
replace name = "coef" in 2
replace name = "se"   in 3
replace value = s_bw   in 1
replace value = s_coef in 2
replace value = s_se   in 3

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\rdd_fuzzy.dta", replace
