*! staggered_did.do — Staggered DiD (SSC: csdid) — balanced cohorts
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv", clear

* Match the Python-side balanced test: keep entities 0-19 only.
* The Python test filters `entity < 20`, which excludes the gvar=5
* (treated-at-t=5) entities 20-29. They never turn on in the data
* (max time = 4), so csdid only ever sees the g=3 cohort after this drop.
drop if entity >= 20

* gvar: 0 = never treated, 3 = treated at time 3
gen gvar = 0
replace gvar = 3 if entity >= 10 & entity < 20

* Run csdid with covariates (dripw, default)
csdid y x z, ivar(entity) time(time) gvar(gvar)

* e(b) is 1x8: first 4 are ATT(g=3, t), last 4 are weights
* (gvar=5 cohort dropped above, so only g=3 remains)
matrix B = e(b)
local nb = colsof(B)
forvalues i = 1/8 {
    local name = ""
    if `i' == 1  local name "b_g3_t0_1"
    if `i' == 2  local name "b_g3_t1_2"
    if `i' == 3  local name "b_g3_t2_3"
    if `i' == 4  local name "b_g3_t2_4"
    if `i' == 5  local name "w_g3_t0_1"
    if `i' == 6  local name "w_g3_t1_2"
    if `i' == 7  local name "w_g3_t2_3"
    if `i' == 8  local name "w_g3_t2_4"
    scalar `name' = B[1,`i']
}

* Store SEs (sqrt of diagonal of e(V), 8x8)
matrix V = e(V)
forvalues i = 1/8 {
    local name = ""
    if `i' == 1  local name "se_g3_t0_1"
    if `i' == 2  local name "se_g3_t1_2"
    if `i' == 3  local name "se_g3_t2_3"
    if `i' == 4  local name "se_g3_t2_4"
    if `i' == 5  local name "se_w_g3_t0_1"
    if `i' == 6  local name "se_w_g3_t1_2"
    if `i' == 7  local name "se_w_g3_t2_3"
    if `i' == 8  local name "se_w_g3_t2_4"
    scalar `name' = sqrt(V[`i',`i'])
}

* Also store e(N)
scalar s_N = e(N)

clear
set obs 17
gen str20 name  = ""
gen double value = .

local i = 1
foreach var in b_g3_t0_1 b_g3_t1_2 b_g3_t2_3 b_g3_t2_4 ///
              w_g3_t0_1 w_g3_t1_2 w_g3_t2_3 w_g3_t2_4 ///
              se_g3_t0_1 se_g3_t1_2 se_g3_t2_3 se_g3_t2_4 ///
              se_w_g3_t0_1 se_w_g3_t1_2 se_w_g3_t2_3 se_w_g3_t2_4 ///
              s_N {
    replace name  = "`var'"  in `i'
    replace value = `var'    in `i'
    local ++i
}

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\staggered_did.dta", replace
