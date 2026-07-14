*! explore_csdid.do — Explore csdid_estat output capture
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear

* gvar: 0 = never treated, 3 = treated at time 3, 5 = treated at time 5
gen gvar = 0
replace gvar = 3 if entity >= 10 & entity < 20
replace gvar = 5 if entity >= 20

* === Step 1: csdid with full DR (default dripw) with covariates ===
noisily display _newline "===== csdid with covariates (default dripw) ====="
csdid y x z, ivar(entity) time(time) gvar(gvar)

* Store e(b) and e(V)
matrix list e(b)
matrix list e(V)

* === Step 2: csdid_estat simple ===
noisily display _newline "===== csdid_estat simple ====="
csdid_estat simple
* Check what's in e() after csdid_estat
capture noisily matrix list e(b)
capture noisily display "e(b) has rows: " rowsof(e(b)) " cols: " colsof(e(b))
capture noisily display "e(b)[1,1] = " e(b)[1,1]

* Try to capture
tempname b_simple
capture noisily matrix `b_simple' = e(b)
capture noisily matrix list `b_simple'

* === Step 3: csdid_estat group ===
noisily display _newline "===== csdid_estat group ====="
csdid_estat group
capture noisily matrix list e(b)
capture noisily display "e(b)[1,1] = " e(b)[1,1]
capture noisily display "e(b)[1,2] = " e(b)[1,2]

* === Step 4: csdid_estat calendar ===
noisily display _newline "===== csdid_estat calendar ====="
csdid_estat calendar
capture noisily matrix list e(b)

* === Step 5: csdid_estat event ===
noisily display _newline "===== csdid_estat event ====="
csdid_estat event
capture noisily matrix list e(b)

* === Step 6: csdid without covariates (method reg) ===
noisily display _newline "===== csdid WITHOUT covariates (auto reg) ====="
csdid y, ivar(entity) time(time) gvar(gvar)
matrix list e(b)

* csdid_estat simple (no covariates)
noisily display _newline "===== csdid_estat simple (no covariates) ====="
csdid_estat simple
capture noisily matrix list e(b)
capture noisily display "e(b)[1,1] = " e(b)[1,1]

* === Step 7: csdid with method(reg) and covariates ===
noisily display _newline "===== csdid with method(reg) and covariates ====="
csdid y x z, ivar(entity) time(time) gvar(gvar), method(reg)
matrix list e(b)

noisily display _newline "===== csdid_estat simple after method(reg) ====="
csdid_estat simple
capture noisily matrix list e(b)
capture noisily display "e(b)[1,1] = " e(b)[1,1]

* === Step 8: csdid with method(drimp) and covariates ===
noisily display _newline "===== csdid method(drimp) with covariates ====="
csdid y x z, ivar(entity) time(time) gvar(gvar), method(drimp)
matrix list e(b)

* === Step 9: csdid without covariates ===
noisily display _newline "===== csdid y, no covariates ====="
csdid y, ivar(entity) time(time) gvar(gvar)
matrix list e(b)

csdid_estat simple
capture noisily matrix list e(b)

csdid_estat event
capture noisily matrix list e(b)

* === Step 10: Test the notyet option ===
noisily display _newline "===== csdid with notyet option ====="
csdid y x z, ivar(entity) time(time) gvar(gvar) notyet
matrix list e(b)

* === Step 11: Try to store results manually ===
noisily display _newline "===== Manual capture attempts ====="
csdid y x z, ivar(entity) time(time) gvar(gvar)
csdid_estat simple

* Try different ways to access the results
capture noisily display "Trying _b[ATT]: " _b[ATT]
capture noisily display "Trying _b[simple]: " _b[simple]
capture noisily display "Trying e(simple): " e(simple)
capture noisily display "Trying e(ATT): " e(ATT)
capture noisily display "Trying e(b) colnames: " colnames(e(b))

* Use return list to see what's stored
noisily display _newline "===== return list after csdid_estat simple ====="
return list

noisily display _newline "===== ereturn list after csdid_estat simple ====="
ereturn list

noisily display _newline "===== creturn list after csdid_estat simple ====="
creturn list

noisily display "Done with exploration."
