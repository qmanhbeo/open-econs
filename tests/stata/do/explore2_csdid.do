*! explore2_csdid.do — Deeper exploration of csdid storage
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv", clear

gen gvar = 0
replace gvar = 3 if entity >= 10 & entity < 20
replace gvar = 5 if entity >= 20

* Run csdid with DR (default) with covariates
csdid y x z, ivar(entity) time(time) gvar(gvar)

* Store the full coefficient vector and weight vector
local k = colsof(e(b))
forvalues i = 1 / `k' {
    local colname = colnames(e(b))[`i']
    noisily display "e(b)[1,`i'] = " el(e(b),1,`i') _col(20) " colname: `colname'"
    local vname = substr("`colname'", 1, 2)
    if "`vname'" != "wg" {
        local att_count = `att_count' + 1
    }
}

* Now run csdid_estat simple and check if anything in ereturn/return is updated
noisily display _newline "=== After csdid_estat simple ==="
estimates store csdid_main

csdid_estat simple

* Check all return values
noisily display _newline "--- ereturn list ---"
ereturn list

noisily display _newline "--- return list ---"
return list

* Check if estimates are stored differently
estimates dir

* Try using csdid_rif
noisily display _newline "=== Try csdid_rif ==="
capture noisily csdid_rif

* Check what matrices are available after csdid_estat
noisily display _newline "--- All matrices ---"
matrix dir

* Now restore original estimates
estimates restore csdid_main

* Try csdid_estat simple, estore(simple_est)
noisily display _newline "=== csdid_estat simple with estore ==="
csdid_estat simple, estore(simple_est)

* Check if estore created estimates
estimates dir
estimates restore simple_est
ereturn list

noisily display _newline "=== Done ==="
