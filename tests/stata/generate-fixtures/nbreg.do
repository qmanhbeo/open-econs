*! Negative binomial regression parity fixture for open-econs
*! Ground truth = Stata SSC nbreg (Hilbe / StataCorp).
*!
*! NOTE (rule 15/16, see methodology/limited/nbreg.md): Stata nbreg's two
*! dispersion settings parameterize NB differently:
*!   - dispersion(mean)   : Var = mu*(1+alpha); its MLE COINCIDES with the
*!     textbook NB2 gamma-mixture (R glm.nb / fixest fenegbin) on this dataset
*!     (mean mu ~ 1). oe.nbreg(dispersion="const") reproduces this exactly.
*!   - dispersion(constant): Var = mu + delta*mu^2; a DIFFERENT MLE (Stata-
*!     specific). oe.nbreg does NOT reproduce it; it is a documented open gap.
*!
*! We record BOTH so the divergence is transparent, but the parity test asserts
*! only dispersion(mean) (== oe nbreg const).

clear all
set more off
set type double
import delimited "tests/r/fixtures/inputs/nbreg_input.csv", clear

* --- dispersion(mean)  (== oe nbreg dispersion="const", NB2 gamma mixture) ---
nbreg y x1 x2, dispersion(mean)
matrix b = e(b)
matrix V = e(V)
scalar b_x1   = b[1,1]
scalar b_x2   = b[1,2]
scalar se_x1  = sqrt(V[1,1])
scalar se_x2  = sqrt(V[2,2])
scalar alpha  = e(alpha)
scalar ll     = e(ll)

* --- dispersion(constant)  (documented Stata-specific divergence) ---
nbreg y x1 x2, dispersion(constant)
matrix bc = e(b)
matrix Vc = e(V)
scalar bc_x1  = bc[1,1]
scalar bc_x2  = bc[1,2]
scalar sec_x1 = sqrt(Vc[1,1])
scalar sec_x2 = sqrt(Vc[2,2])
scalar delta  = e(delta)    // Stata stores NB2 overdispersion as delta
scalar llc    = e(ll)

* --- build name/value dataset ---
clear
set obs 20
gen str32 name = ""
gen double value = .

replace name = "b_x1"   in 1
replace value = b_x1    in 1
replace name = "b_x2"   in 2
replace value = b_x2    in 2
replace name = "se_x1"  in 3
replace value = se_x1   in 3
replace name = "se_x2"  in 4
replace value = se_x2   in 4
replace name = "alpha"  in 5
replace value = alpha   in 5
replace name = "ll"     in 6
replace value = ll      in 6
replace name = "bc_x1"  in 7
replace value = bc_x1   in 7
replace name = "bc_x2"  in 8
replace value = bc_x2   in 8
replace name = "sec_x1" in 9
replace value = sec_x1  in 9
replace name = "sec_x2" in 10
replace value = sec_x2  in 10
replace name = "delta"  in 11
replace value = delta   in 11
replace name = "llc"    in 12
replace value = llc     in 12

drop if name == ""
save "tests/stata/fixtures/expected/nbreg.dta", replace
