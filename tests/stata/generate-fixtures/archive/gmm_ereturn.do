clear all
set more off
set logtype text
log using "C:\Users\manhn\AppData\Local\Temp\gmm_ereturn.log", replace
import delimited "C:\Users\manhn\Desktop\open-econs\tests/stata/fixtures/inputs/df_gmm.csv", clear

gmm (y - {b0} - {b1}*x1 - {b2}*x2), instruments(z1 z2 z3 z4 z5) ///
    winitial(unadjusted) twostep vce(robust)

ereturn list
log close
