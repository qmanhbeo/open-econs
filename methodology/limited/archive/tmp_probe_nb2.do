clear all
set more off
set type double
import delimited "tests/r/fixtures/inputs/nbreg_input.csv", clear
nbreg y x1 x2, dispersion(constant)
disp "alpha=" e(alpha)
disp "lnalpha_disp=" e(lnalpha)
scalar ln2 = ln(e(alpha))
disp "ln(alpha)=" ln2
nbreg y x1 x2, dispersion(mean)
disp "alpha1=" e(alpha)
disp "lnalpha1_disp=" e(lnalpha)
scalar ln1 = ln(e(alpha))
disp "ln(alpha1)=" ln1
log using "tmp_nb_log.txt", replace text
disp "NB2 ln(alpha)=" ln2
disp "NB1 ln(alpha)=" ln1
disp "NB2 alpha=" e(alpha)
disp "NB1 alpha=" e(alpha)
log close
