*! explore3_csdid.do — Inspect csdid internals for IF extraction
clear all
set more off
log using "C:\Users\manhn\Desktop\open-econs\tests\stata\do\explore3_csdid.log", replace text

import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv", clear
drop if entity >= 20
gen gvar = 0
replace gvar = 3 if entity >= 10 & entity < 20

csdid y x z, ivar(entity) time(time) gvar(gvar)

di as text _newline "===== ereturn list ====="
ereturn list
di as text _newline "===== matrix dir ====="
matrix dir
di as text _newline "===== e(b) ====="
matrix list e(b)
di as text _newline "===== e(V) ====="
matrix list e(V)

di as text _newline "===== try csdid_rif ====="
capture noisily csdid_rif
di as text "rc after csdid_rif = " _rc

di as text _newline "===== ereturn list after csdid_rif ====="
capture ereturn list
di as text _newline "===== matrix dir after csdid_rif ====="
capture matrix dir

di as text _newline "===== check e(rif) / e(inf) ====="
capture confirm matrix e(rif)
di as text "e(rif) exists: " (_rc==0)
capture confirm matrix e(inf)
di as text "e(inf) exists: " (_rc==0)
capture confirm matrix e(IF)
di as text "e(IF) exists: " (_rc==0)

di as text "DONE"
log close
