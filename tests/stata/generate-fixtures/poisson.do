*! Poisson FE (PPML) parity fixture for open-econs
*! Ground truth = Stata SSC ppmlhdfe (Correia-Guimaraes-Zylkin 2020).
*!
*! Records, for the canonical two-way FE count panel (y ~ x1 x2 | firm year):
*!   - cluster(firm) SEs  (default ppmlhdfe, matches oe.poisson vcov_backend="stata")
*!   - nonrobust (iid) SEs (matches oe.poisson cov_type="nonrobust",
*!     vcov_backend="stata")
*!   - coefficient point estimates, deviance, log pseudolikelihood.
*!
*! ROOT CAUSE (rule 16, see methodology/limited/poisson.md): ppmlhdfe applies
*! G_adj=G/(G-1) but NOT k_adj=(N-1)/(N-K), and treats FE nested in the cluster
*! as redundant (0 dof). fixest/pyfixest default the opposite. oe.poisson
*! vcov_backend="stata" reproduces ppmlhdfe via ssc(k_adj=False,G_adj=True,
*! k_fixef="none"). These fixtures are the ppmlhdfe numbers.

clear all
set more off
set type double

import delimited "tests/r/fixtures/inputs/poisson_input.csv", clear
assert y >= 0

* --- cluster(firm) SEs (default) ---
ppmlhdfe y x1 x2, absorb(firm year) cluster(firm)
matrix b = e(b)
matrix V = e(V)
scalar b_x1 = b[1,1]
scalar b_x2 = b[1,2]
scalar se_x1_clu = sqrt(V[1,1])
scalar se_x2_clu = sqrt(V[2,2])
scalar ll_clu = e(ll)
scalar dev_clu = e(deviance)       // ppmlhdfe stores deviance in e(deviance)

* --- nonrobust (iid) SEs ---
ppmlhdfe y x1 x2, absorb(firm year)
matrix V0 = e(V)
scalar se_x1_iid = sqrt(V0[1,1])
scalar se_x2_iid = sqrt(V0[2,2])

* --- build name/value dataset ---
clear
set obs 20
gen str32 name = ""
gen double value = .

replace name = "b_x1"      in 1
replace value = b_x1       in 1
replace name = "b_x2"      in 2
replace value = b_x2       in 2
replace name = "se_x1_clu" in 3
replace value = se_x1_clu  in 3
replace name = "se_x2_clu" in 4
replace value = se_x2_clu  in 4
replace name = "ll_clu"    in 5
replace value = ll_clu     in 5
replace name = "dev_clu"   in 6
replace value = dev_clu    in 6
replace name = "se_x1_iid" in 7
replace value = se_x1_iid  in 7
replace name = "se_x2_iid" in 8
replace value = se_x2_iid  in 8

drop if name == ""
save "tests/stata/fixtures/expected/poisson.dta", replace
