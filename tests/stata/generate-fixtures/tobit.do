*! Tobit (censored normal) MLE parity fixture for open-econs
*! Ground truth = Stata base tobit.
*!
*! Records, for both a left-censored (ll(0)) and a no-censoring variant:
*!   - coefficient point estimates
*!   - sigma  (Stata reports Log(scale) = ln sigma in the header; sigma =
*!     exp(e(lnlnsigma))). We store BOTH sigma and log_scale so OE can be
*!     cross-checked against Stata's printed header AND R's sigma directly.
*!   - nonrobust (OIM) SEs
*!   - log-likelihood  (e(ll))
*!   - left-censored count (e(N_lcens))
*!
*! ROOT CAUSE (rule 16, see methodology/limited/tobit.md): Stata tobit reports
*! Log(scale) = ln(sigma), while R AER::tobit reports sigma directly. Both carry
*! the SAME covariance for the regressors; the (k+1) OIM includes a sigma row.
*! OE reports sigma, log_scale, and the regressor-only table (Stata convention).

clear all
set more off
set type double

import delimited "tests/r/fixtures/inputs/tobit_input.csv", clear

* --- left-censored Tobit at 0 (ll(0)) ---
tobit y_left x1 x2 x3, ll(0)
matrix b = e(b)
matrix V = e(V)
scalar b_x1 = b[1,1]
scalar b_x2 = b[1,2]
scalar b_x3 = b[1,3]
scalar se_x1 = sqrt(V[1,1])
scalar se_x2 = sqrt(V[2,2])
scalar se_x3 = sqrt(V[3,3])
scalar sigma_l = sqrt(b[1,5])          // Stata reports var(e.y)=sigma^2 as 5th e(b) elt
scalar logscale_l = 0.5 * ln(b[1,5])    // ln(sigma) = 0.5 * ln(sigma^2)
scalar ll_l = e(ll)
scalar n_left_l = e(N_lc)
scalar n_right_l = e(N_rc)

* --- robust (vce(robust)) and cluster (vce(cluster id)) SEs (left-censored) ---
tobit y_left x1 x2 x3, ll(0) vce(robust)
scalar rse_x1 = _se[x1]
scalar rse_x2 = _se[x2]
scalar rse_x3 = _se[x3]
tobit y_left x1 x2 x3, ll(0) vce(cluster id)
scalar cse_x1 = _se[x1]
scalar cse_x2 = _se[x2]
scalar cse_x3 = _se[x3]

* --- no-censoring variant (left(-inf) right(inf)): OLS-equivalent MLE ---
tobit y_nocens x1 x2 x3, ll(-1e15) ul(1e15)
matrix b2 = e(b)
matrix V2 = e(V)
scalar b2_x1 = b2[1,1]
scalar b2_x2 = b2[1,2]
scalar b2_x3 = b2[1,3]
scalar se2_x1 = sqrt(V2[1,1])
scalar se2_x2 = sqrt(V2[2,2])
scalar se2_x3 = sqrt(V2[3,3])
scalar sigma2 = sqrt(b2[1,5])          // Stata reports var(e.y)=sigma^2 as 5th e(b) elt
scalar ll2 = e(ll)

* --- build name/value dataset ---
clear
set obs 32
gen str32 name = ""
gen double value = .

local i = 1
foreach v in b_x1 b_x2 b_x3 se_x1 se_x2 se_x3 sigma_l logscale_l ll_l n_left_l n_right_l ///
               rse_x1 rse_x2 rse_x3 cse_x1 cse_x2 cse_x3 ///
               b2_x1 b2_x2 b2_x3 se2_x1 se2_x2 se2_x3 sigma2 ll2 {
    replace name = "`v'" in `i'
    replace value = `v' in `i'
    local ++i
}

drop if name == ""
save "tests/stata/fixtures/expected/tobit.dta", replace
