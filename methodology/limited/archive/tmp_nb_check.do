clear all
set more off
set type double
import delimited "C:/Users/manhn/Desktop/open-econs/tests/r/fixtures/inputs/nbreg_input.csv", clear
log using "C:/Users/manhn/Desktop/open-econs/tmp_nb_check2.txt", replace text

nbreg y x1 x2, dispersion(constant)
matrix list e(b)
scalar _b2_x1 = _b[x1]
scalar _b2_x2 = _b[x2]
scalar _a2 = e(alpha)
scalar _ll2 = e(ll)
disp "NB2: b1=" _b2_x1 " b2=" _b2_x2 " alpha=" _a2 " ll=" _ll2

nbreg y x1 x2, dispersion(mean)
scalar _b1_x1 = _b[x1]
scalar _b1_x2 = _b[x2]
scalar _a1 = e(alpha)
scalar _ll1 = e(ll)
disp "NB1: b1=" _b1_x1 " b2=" _b1_x2 " alpha=" _a1 " ll=" _ll1
log close
