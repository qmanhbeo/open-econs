*! rdd_sharp.do — Sharp RDD (SSC: rdrobust)
clear all
set more off
capture ssc install rdrobust
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_rdd.csv", clear
rdrobust y_sharp x, c(0)

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

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\rdd_sharp.dta", replace
