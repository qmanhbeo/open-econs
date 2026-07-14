*! psm_basic.do — PSM 1:1 with replacement, ATE, logit PS, no effective caliper.
*! teffects psmatch only supports matching WITH replacement (no noreplacement opt).
*! caliper(1.0) removes the caliper constraint (PS in [0,1]).
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_psm.csv", clear

teffects psmatch (y) (t x1 x2, logit), ate caliper(1.0)

scalar s_ate   = el(e(b), 1, 1)
scalar s_se    = sqrt(el(e(V), 1, 1))
scalar s_N     = e(N)

count if t == 1
scalar s_n_t = r(N)
count if t == 0
scalar s_n_c = r(N)

* min/max matches per observation
tempname mat
mat `mat' = e(matches)
scalar s_min_match = el(`mat', 1, 1)
scalar s_max_match = el(`mat', 1, 2)

clear
set obs 8
gen str20 name  = ""
gen double value = .
replace name = "ate"        in 1
replace name = "se"         in 2
replace name = "N"          in 3
replace name = "n_t"        in 4
replace name = "n_c"        in 5
replace name = "min_match"  in 6
replace name = "max_match"  in 7
replace name = "variance"   in 8
replace value = s_ate        in 1
replace value = s_se         in 2
replace value = s_N          in 3
replace value = s_n_t        in 4
replace value = s_n_c        in 5
replace value = s_min_match  in 6
replace value = s_max_match  in 7
replace value = el(e(V), 1, 1) in 8

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\psm_basic.dta", replace
