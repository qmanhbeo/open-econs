*! rdd_density.do — McCrary/CJM density (manipulation) test (SSC: rddensity)
clear all
set more off
capture ssc install rddensity
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_rdd_density.csv", clear

* Manipulation test at cutoff 0 (density discontinuity)
rddensity x, c(0)

* theta = fhat_r - fhat_l (q-order density estimates, the test basis)
scalar s_theta   = e(f_qr) - e(f_ql)
scalar s_se      = e(se_q)
scalar s_z       = e(T_q)
scalar s_p       = e(pv_q)
scalar s_h_l     = e(h_l)
scalar s_h_r     = e(h_r)
scalar s_n_l     = e(N_l)
scalar s_n_r     = e(N_r)
scalar s_fhat_l  = e(f_ql)
scalar s_fhat_r  = e(f_qr)
* p-order estimates (diagnostic: which order does the Python wrapper report?)
scalar s_fhat_l_p = e(f_pl)
scalar s_fhat_r_p = e(f_pr)

clear
set obs 12
gen str20 name  = ""
gen double value = .
replace name = "theta"      in 1
replace name = "se"         in 2
replace name = "z"          in 3
replace name = "p"          in 4
replace name = "h_l"        in 5
replace name = "h_r"        in 6
replace name = "n_l"        in 7
replace name = "n_r"        in 8
replace name = "fhat_l"     in 9
replace name = "fhat_r"     in 10
replace name = "fhat_l_p"   in 11
replace name = "fhat_r_p"   in 12
replace value = s_theta     in 1
replace value = s_se        in 2
replace value = s_z         in 3
replace value = s_p         in 4
replace value = s_h_l       in 5
replace value = s_h_r       in 6
replace value = s_n_l       in 7
replace value = s_n_r       in 8
replace value = s_fhat_l    in 9
replace value = s_fhat_r    in 10
replace value = s_fhat_l_p  in 11
replace value = s_fhat_r_p  in 12

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\rdd_density.dta", replace
