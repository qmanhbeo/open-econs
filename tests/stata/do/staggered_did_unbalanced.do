*! staggered_did_unbalanced.do — Staggered DiD with unbalanced cohorts (SSC: csdid)
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel_unbalanced.csv", clear

* gvar: 0 = never treated, 3 = treated at time 3, 5 = treated at time 5
* entity 0-14: never treated (15)
* entity 15-22: treated at t=3 (8)
* entity 23-29: treated at t=5 (7)
gen gvar = 0
replace gvar = 3 if entity >= 15 & entity < 23
replace gvar = 5 if entity >= 23

* Run csdid with covariates (dripw, default)
csdid y x z, ivar(entity) time(time) gvar(gvar)

* Store all 8 ATT(g,t) coefficients and 8 weights from e(b)
* e(b) is 1x16: first 8 are ATT(g,t), last 8 are weights
forvalues i = 1/16 {
    local name = ""
    if `i' == 1  local name "b_g3_t0_1"
    if `i' == 2  local name "b_g3_t1_2"
    if `i' == 3  local name "b_g3_t2_3"
    if `i' == 4  local name "b_g3_t2_4"
    if `i' == 5  local name "b_g5_t0_1"
    if `i' == 6  local name "b_g5_t1_2"
    if `i' == 7  local name "b_g5_t2_3"
    if `i' == 8  local name "b_g5_t3_4"
    if `i' == 9  local name "w_g3_t0_1"
    if `i' == 10 local name "w_g3_t1_2"
    if `i' == 11 local name "w_g3_t2_3"
    if `i' == 12 local name "w_g3_t2_4"
    if `i' == 13 local name "w_g5_t0_1"
    if `i' == 14 local name "w_g5_t1_2"
    if `i' == 15 local name "w_g5_t2_3"
    if `i' == 16 local name "w_g5_t3_4"
    scalar `name' = e(b)[1,`i']
}

* Store SEs (sqrt of diagonal of e(V), 16x16)
forvalues i = 1/16 {
    local name = ""
    if `i' == 1  local name "se_g3_t0_1"
    if `i' == 2  local name "se_g3_t1_2"
    if `i' == 3  local name "se_g3_t2_3"
    if `i' == 4  local name "se_g3_t2_4"
    if `i' == 5  local name "se_g5_t0_1"
    if `i' == 6  local name "se_g5_t1_2"
    if `i' == 7  local name "se_g5_t2_3"
    if `i' == 8  local name "se_g5_t3_4"
    if `i' == 9  local name "se_w_g3_t0_1"
    if `i' == 10 local name "se_w_g3_t1_2"
    if `i' == 11 local name "se_w_g3_t2_3"
    if `i' == 12 local name "se_w_g3_t2_4"
    if `i' == 13 local name "se_w_g5_t0_1"
    if `i' == 14 local name "se_w_g5_t1_2"
    if `i' == 15 local name "se_w_g5_t2_3"
    if `i' == 16 local name "se_w_g5_t3_4"
    scalar `name' = sqrt(e(V)[`i',`i'])
}

* Also store e(N)
scalar s_N = e(N)

clear
set obs 33
gen str20 name  = ""
gen double value = .

local i = 1
foreach var in b_g3_t0_1 b_g3_t1_2 b_g3_t2_3 b_g3_t2_4 b_g5_t0_1 b_g5_t1_2 b_g5_t2_3 b_g5_t3_4 ///
              w_g3_t0_1 w_g3_t1_2 w_g3_t2_3 w_g3_t2_4 w_g5_t0_1 w_g5_t1_2 w_g5_t2_3 w_g5_t3_4 ///
              se_g3_t0_1 se_g3_t1_2 se_g3_t2_3 se_g3_t2_4 se_g5_t0_1 se_g5_t1_2 se_g5_t2_3 se_g5_t3_4 ///
              se_w_g3_t0_1 se_w_g3_t1_2 se_w_g3_t2_3 se_w_g3_t2_4 se_w_g5_t0_1 se_w_g5_t1_2 se_w_g5_t2_3 se_w_g5_t3_4 ///
              s_N {
    replace name  = "`var'"  in `i'
    replace value = `var'    in `i'
    local ++i
}

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\staggered_did_unbalanced.dta", replace
