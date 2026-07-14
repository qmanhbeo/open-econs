*! rosenbaum_pairs.do -- Rosenbaum bounds on a small controlled pair-diff
*! dataset that deliberately includes zero-difference pairs.
clear all
set more off

* 10 matched pairs with known differences.
* Includes TWO zero-difference pairs (obs 4 and 9) to exercise the
* zero-handling convention.
set obs 10
gen double diff = .
replace diff =  1.5 in 1
replace diff =  2.0 in 2
replace diff = -0.5 in 3
replace diff =  0.0 in 4
replace diff =  3.0 in 5
replace diff = -1.0 in 6
replace diff =  0.5 in 7
replace diff = -2.0 in 8
replace diff =  0.0 in 9
replace diff =  1.0 in 10

rbounds diff, gamma(1 2 3)
mat R = r(outmat)

* R cols: gamma, sig+, sig-, t-hat+, t-hat-, CI+, CI-
scalar n_pairs   = r(N)
scalar g1_sigp   = R[1, 2]
scalar g1_sigm   = R[1, 3]
scalar g2_sigp   = R[2, 2]
scalar g2_sigm   = R[2, 3]
scalar g3_sigp   = R[3, 2]
scalar g3_sigm   = R[3, 3]

clear
set obs 7
gen str20 name   = ""
gen double value = .
replace name = "N"        in 1
replace name = "g1_sigp"  in 2
replace name = "g1_sigm"  in 3
replace name = "g2_sigp"  in 4
replace name = "g2_sigm"  in 5
replace name = "g3_sigp"  in 6
replace name = "g3_sigm"  in 7
replace value = n_pairs   in 1
replace value = g1_sigp  in 2
replace value = g1_sigm  in 3
replace value = g2_sigp  in 4
replace value = g2_sigm  in 5
replace value = g3_sigp  in 6
replace value = g3_sigm  in 7

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\rosenbaum_pairs.dta", replace
