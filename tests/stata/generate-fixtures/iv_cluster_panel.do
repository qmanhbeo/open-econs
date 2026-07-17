*! iv_cluster_panel.do — IV/2SLS overidentified, cluster-robust SEs (Stata parity)
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_iv_panel.csv", clear
ivregress 2sls y w (x = z1 z2), vce(cluster id)
scalar s_b0  = _b[_cons]
scalar s_bw  = _b[w]
scalar s_bx  = _b[x]
scalar s_se0 = _se[_cons]
scalar s_sew = _se[w]
scalar s_sex = _se[x]

clear
set obs 6
gen str20 name = ""
gen double value = .
replace name = "b_int" in 1
replace value = s_b0 in 1
replace name = "b_w" in 2
replace value = s_bw in 2
replace name = "b_x" in 3
replace value = s_bx in 3
replace name = "se_int" in 4
replace value = s_se0 in 4
replace name = "se_w" in 5
replace value = s_sew in 5
replace name = "se_x" in 6
replace value = s_sex in 6

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\iv_cluster_panel.dta", replace
