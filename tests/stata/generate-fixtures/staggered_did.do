*! staggered_did.do — Staggered DiD (SSC: csdid) — balanced cohorts
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear

* Match the Python-side balanced test: keep entities 0-19 only.
* The Python test filters `entity < 20`, which excludes the gvar=5
* (treated-at-t=5) entities 20-29. They never turn on in the data
* (max time = 4), so csdid only ever sees the g=3 cohort after this drop.
drop if entity >= 20

* gvar: 0 = never treated, 3 = treated at time 3
gen gvar = 0
replace gvar = 3 if entity >= 10 & entity < 20

* Run csdid with covariates (dripw, default).  saverif() dumps the per-entity
* RIF dataset so we can recompute the full-sample aggregated SE the same way
* open_econs does (equal-weight average over post-treatment cells, then the
* cluster-robust influence-function variance V = sum((RIF-mean)^2)/N^2).
csdid y x z, ivar(entity) time(time) gvar(gvar) saverif(tmp_rif_bal)

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

* ---- Aggregated SE from per-entity RIF (open_econs method) ----
use tmp_rif_bal.dta, clear
* Post-treatment cells for cohort g=3 are t=3 (_g3_2_3) and t=4 (_g3_2_4).
gen double agg_rif = (_g3_2_3 + _g3_2_4) / 2
summ agg_rif
scalar mean_r = r(mean)
gen double dev = agg_rif - mean_r
egen double ssd = sum(dev^2)
scalar agg_se = sqrt(ssd / (_N^2))
erase tmp_rif_bal.dta
drop _all

clear
set obs 18
gen str20 name  = ""
gen double value = .

replace name  = "b_g3_t0_1"      in 1
replace value = b_g3_t0_1        in 1
replace name  = "b_g3_t1_2"      in 2
replace value = b_g3_t1_2        in 2
replace name  = "b_g3_t2_3"      in 3
replace value = b_g3_t2_3        in 3
replace name  = "b_g3_t2_4"      in 4
replace value = b_g3_t2_4        in 4
replace name  = "w_g3_t0_1"      in 5
replace value = w_g3_t0_1        in 5
replace name  = "w_g3_t1_2"      in 6
replace value = w_g3_t1_2        in 6
replace name  = "w_g3_t2_3"      in 7
replace value = w_g3_t2_3        in 7
replace name  = "w_g3_t2_4"      in 8
replace value = w_g3_t2_4        in 8
replace name  = "se_g3_t0_1"     in 9
replace value = se_g3_t0_1       in 9
replace name  = "se_g3_t1_2"     in 10
replace value = se_g3_t1_2       in 10
replace name  = "se_g3_t2_3"     in 11
replace value = se_g3_t2_3       in 11
replace name  = "se_g3_t2_4"     in 12
replace value = se_g3_t2_4       in 12
replace name  = "se_w_g3_t0_1"   in 13
replace value = se_w_g3_t0_1     in 13
replace name  = "se_w_g3_t1_2"   in 14
replace value = se_w_g3_t1_2     in 14
replace name  = "se_w_g3_t2_3"   in 15
replace value = se_w_g3_t2_3     in 15
replace name  = "se_w_g3_t2_4"   in 16
replace value = se_w_g3_t2_4     in 16
replace name  = "s_N"            in 17
replace value = s_N              in 17
replace name  = "agg_se"         in 18
replace value = agg_se           in 18

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\staggered_did.dta", replace
