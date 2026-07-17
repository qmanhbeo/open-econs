clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_iv_panel.csv", clear
xtset id t

* (A) xtivreg fe vce(robust)
xtivreg y w (x = z1 z2), fe vce(robust)
scalar s_xtiv_rob_sew = _se[w]
scalar s_xtiv_rob_dfr = e(df_rz)

* (B) xtivreg fe nonrobust
xtivreg y w (x = z1 z2), fe
scalar s_xtiv_nr_sew = _se[w]
scalar s_xtiv_nr_dfr = e(df_rz)

* (C) manual within + _regress , cluster(id)  (same as xtivreg robust inner)
xtdata y w x z1 z2, i(id) fe clear
_regress y w x (w z1 z2), cluster(id)
scalar s_inner_cl_sew = _se[w]
scalar s_inner_cl_dfr = e(df_r)

clear
set obs 6
gen str20 name=""
gen double value=.
replace name="xtiv_rob_sew" in 1
replace value=s_xtiv_rob_sew in 1
replace name="xtiv_rob_dfr" in 2
replace value=s_xtiv_rob_dfr in 2
replace name="xtiv_nr_sew" in 3
replace value=s_xtiv_nr_sew in 3
replace name="xtiv_nr_dfr" in 4
replace value=s_xtiv_nr_dfr in 4
replace name="inner_cl_sew" in 5
replace value=s_inner_cl_sew in 5
replace name="inner_cl_dfr" in 6
replace value=s_inner_cl_dfr in 6
save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\iv_fe_diag.dta", replace