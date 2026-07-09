*! staggered_did.do — Staggered DiD (SSC: csdid)
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv", clear

* gvar: 0 = never treated, 3 = treated at time 3, 5 = treated at time 5
gen gvar = 0
replace gvar = 3 if entity >= 10 & entity < 20
replace gvar = 5 if entity >= 20

csdid y x z, ivar(entity) time(time) gvar(gvar)

scalar s_att = e(b)[1,1]
scalar s_se  = sqrt(e(V)[1,1])

clear
set obs 2
gen str20 name  = ""
gen double value = .
replace name = "ATT" in 1
replace name = "se"  in 2
replace value = s_att in 1
replace value = s_se  in 2

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\staggered_did.dta", replace
