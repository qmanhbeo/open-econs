*! NB regression probe (pooled) for open-econs
clear all
set more off
set type double
import delimited "tests/r/fixtures/inputs/nbreg_input.csv", clear

* NB2 (default), dispersion(constant)
nbreg y x1 x2, dispersion(constant)
matrix b = e(b)
matrix V = e(V)
scalar b2_x1 = b[1,1]
scalar b2_x2 = b[1,2]
scalar se2_x1 = sqrt(V[1,1])
scalar se2_x2 = sqrt(V[2,2])
scalar ln2 = e(lnalpha)
scalar a2 = e(alpha)
scalar ll2 = e(ll)
scalar ll2_c = e(ll_c)
disp "NB2 lnalpha=" ln2 " alpha=" a2
disp "NB2 se x1=" se2_x1 " se x2=" se2_x2

* NB1, dispersion(mean)
nbreg y x1 x2, dispersion(mean)
matrix b1 = e(b)
matrix V1 = e(V)
scalar b1_x1 = b1[1,1]
scalar b1_x2 = b1[1,2]
scalar se1_x1 = sqrt(V1[1,1])
scalar se1_x2 = sqrt(V1[2,2])
scalar ln1 = e(lnalpha)
scalar a1 = e(alpha)
scalar ll1 = e(ll)
disp "NB1 lnalpha=" ln1 " alpha=" a1

clear
set obs 16
gen str32 name = ""
gen double value = .
replace name = "b2_x1" in 1
replace value = b2_x1 in 1
replace name = "b2_x2" in 2
replace value = b2_x2 in 2
replace name = "se2_x1" in 3
replace value = se2_x1 in 3
replace name = "se2_x2" in 4
replace value = se2_x2 in 4
replace name = "lnalpha2" in 5
replace value = ln2 in 5
replace name = "alpha2" in 6
replace value = a2 in 6
replace name = "ll2" in 7
replace value = ll2 in 7
replace name = "ll2_c" in 8
replace value = ll2_c in 8
replace name = "b1_x1" in 9
replace value = b1_x1 in 9
replace name = "b1_x2" in 10
replace value = b1_x2 in 10
replace name = "se1_x1" in 11
replace value = se1_x1 in 11
replace name = "se1_x2" in 12
replace value = se1_x2 in 12
replace name = "lnalpha1" in 13
replace value = ln1 in 13
replace name = "alpha1" in 14
replace value = a1 in 14
replace name = "ll1" in 15
replace value = ll1 in 15
drop if name == ""
save "tmp_nb_stata.dta", replace
