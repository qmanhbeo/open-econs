*! did_cluster.do â€” DiD with cluster SEs
clear all
set more off
set type double
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_did.csv", clear
gen treat_post = treat * post
regress y treat post treat_post, cluster(unit)

scalar s_se_did = _se[treat_post]

clear
set obs 1
gen str20 name  = ""
gen double value = .
replace name = "se_treatXpost" in 1
replace value = s_se_did in 1

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\did_cluster.dta", replace
