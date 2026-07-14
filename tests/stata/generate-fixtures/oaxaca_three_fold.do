*! oaxaca_three_fold.do — Oaxaca-Blinder three-fold (default + reverse)
*! (SSC: oaxaca v4.1.1, Ben Jann)
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_oaxaca.csv", clear

* --- three-fold default (group 2 coefficients as reference) ---
oaxaca y edu age, by(female)
matrix b_3f = e(b)
scalar threefold_gap          = b_3f[1,3]
scalar threefold_endowment    = b_3f[1,4]
scalar threefold_coefficients = b_3f[1,5]
scalar threefold_interaction  = b_3f[1,6]

* --- three-fold reverse (group 1 coefficients as reference) ---
oaxaca y edu age, by(female) threefold(reverse)
matrix b_3fr = e(b)
scalar threefold_rev_gap          = b_3fr[1,3]
scalar threefold_rev_endowment    = b_3fr[1,4]
scalar threefold_rev_coefficients = b_3fr[1,5]
scalar threefold_rev_interaction  = b_3fr[1,6]

* --- assemble reference table ---
clear
set obs 8
gen str30 name  = ""
gen double value = .

local s threefold_gap
replace name = "`s'" in 1
replace value = `s' in 1

local s threefold_endowment
replace name = "`s'" in 2
replace value = `s' in 2

local s threefold_coefficients
replace name = "`s'" in 3
replace value = `s' in 3

local s threefold_interaction
replace name = "`s'" in 4
replace value = `s' in 4

local s threefold_rev_gap
replace name = "`s'" in 5
replace value = `s' in 5

local s threefold_rev_endowment
replace name = "`s'" in 6
replace value = `s' in 6

local s threefold_rev_coefficients
replace name = "`s'" in 7
replace value = `s' in 7

local s threefold_rev_interaction
replace name = "`s'" in 8
replace value = `s' in 8

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\oaxaca_three_fold.dta", replace
