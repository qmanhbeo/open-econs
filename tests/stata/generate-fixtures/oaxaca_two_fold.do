*! oaxaca_two_fold.do â€” Oaxaca-Blinder two-fold via pooled, omega, weight(1), weight(0)
*! (SSC: oaxaca v4.1.1, Ben Jann)
clear all
set more off
set type double
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_oaxaca.csv", clear

* --- pooled reference ---
oaxaca y edu age, by(female) pooled
matrix b_pooled = e(b)
scalar pooled_gap        = b_pooled[1,3]
scalar pooled_explained  = b_pooled[1,4]
scalar pooled_unexplained = b_pooled[1,5]

* --- omega reference (Neumark, no group dummy) ---
oaxaca y edu age, by(female) omega
matrix b_omega = e(b)
scalar omega_gap        = b_omega[1,3]
scalar omega_explained  = b_omega[1,4]
scalar omega_unexplained = b_omega[1,5]

* --- weight(1) reference (group 1 coefficients) ---
oaxaca y edu age, by(female) weight(1)
matrix b_w1 = e(b)
scalar weight1_gap        = b_w1[1,3]
scalar weight1_explained  = b_w1[1,4]
scalar weight1_unexplained = b_w1[1,5]

* --- weight(0) reference (group 2 coefficients) ---
oaxaca y edu age, by(female) weight(0)
matrix b_w0 = e(b)
scalar weight0_gap        = b_w0[1,3]
scalar weight0_explained  = b_w0[1,4]
scalar weight0_unexplained = b_w0[1,5]

* --- assemble reference table ---
clear
set obs 12
gen str30 name  = ""
gen double value = .

local s pooled_gap
replace name = "`s'" in 1
replace value = `s' in 1

local s pooled_explained
replace name = "`s'" in 2
replace value = `s' in 2

local s pooled_unexplained
replace name = "`s'" in 3
replace value = `s' in 3

local s omega_gap
replace name = "`s'" in 4
replace value = `s' in 4

local s omega_explained
replace name = "`s'" in 5
replace value = `s' in 5

local s omega_unexplained
replace name = "`s'" in 6
replace value = `s' in 6

local s weight1_gap
replace name = "`s'" in 7
replace value = `s' in 7

local s weight1_explained
replace name = "`s'" in 8
replace value = `s' in 8

local s weight1_unexplained
replace name = "`s'" in 9
replace value = `s' in 9

local s weight0_gap
replace name = "`s'" in 10
replace value = `s' in 10

local s weight0_explained
replace name = "`s'" in 11
replace value = `s' in 11

local s weight0_unexplained
replace name = "`s'" in 12
replace value = `s' in 12

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\oaxaca_two_fold.dta", replace
