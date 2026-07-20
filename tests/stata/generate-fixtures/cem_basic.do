*! cem_basic.do â€” CEM with explicit cutpoints, export per-observation results.
*!
*! Cutpoints:
*!   x1: 5 equally-spaced breakpoints (-2, -1, 0, 1, 2) -> 4 bins
*!   x2: single cutpoint 0.5 -> 2 bins
*!   x3: exact matching (#0 = zero breakpoints)

clear all
set more off

set type double
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_cem.csv", clear

* Display data summary
sum x1, detail
local x1_min = r(min)
local x1_max = r(max)
dis "x1 range: [`x1_min', `x1_max']"

* Run CEM with explicit cutpoints
cem x1(-2 -1 0 1 2) x2(0.5) x3(#0), treatment(t) showbreaks

* Save full dataset with CEM variables
save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\cem_basic.dta", replace

* Also save scalars for quick reference
egen double N_T_matched = total(t & cem_matched)
egen double N_C_matched = total((1-t) & cem_matched)
egen double sum_w_t = total(t * cem_weights)
egen double sum_w_c = total((1-t) * cem_weights)
egen double n_strata = max(cem_strata)
egen double n_matched_strata = total(cem_matched == 1) if cem_matched == 1
egen double n_matched_strata_u = max(cem_strata * cem_matched)
local n_ms = n_matched_strata_u[1]

scalar s_N_T_matched = N_T_matched[1]
scalar s_N_C_matched = N_C_matched[1]
scalar s_sum_w_t = sum_w_t[1]
scalar s_sum_w_c = sum_w_c[1]
scalar s_n_strata = n_strata[1]
scalar s_n_ms = `n_ms'

clear
set obs 6
gen str32 name = ""
gen double value = .

replace name = "N_T_matched"   in 1
replace name = "N_C_matched"   in 2
replace name = "sum_w_t"       in 3
replace name = "sum_w_c"       in 4
replace name = "n_strata"      in 5
replace name = "n_mstrata"     in 6

replace value = s_N_T_matched  in 1
replace value = s_N_C_matched  in 2
replace value = s_sum_w_t      in 3
replace value = s_sum_w_c      in 4
replace value = s_n_strata     in 5
replace value = s_n_ms         in 6

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\cem_basic_scalars.dta", replace

dis _newline(1) "--- CEM Summary ---"
dis "N_T_matched = " s_N_T_matched
dis "N_C_matched = " s_N_C_matched
dis "sum(w_t)    = " s_sum_w_t
dis "sum(w_c)    = " s_sum_w_c
dis "n_strata    = " s_n_strata
dis "n_mstrata   = " s_n_ms
