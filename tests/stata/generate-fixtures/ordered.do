*! Ordered logit/probit parity fixture for open-econs
*! Ground truth = Stata base ologit / oprobit (dropping the FIRST category as
*! the reference, matching R MASS::polr default).
*!
*! Records, for the canonical ordered model (y ~ x1 x2 x3, 4 categories):
*!   - ologit: coefficients (x1,x2,x3), cutpoints (cut1,cut2,cut3), SEs, ll
*!   - oprobit: same structure with the probit link
*!
*! ROOT CAUSE (rule 16, see methodology/limited/ordered.md): Stata stores
*! cutpoints with P(Y<=j) = F(x'b - cut_j); R MASS::polr stores c_j with
*! P(Y<=j) = F(c_j - x'b). Stata's cut_j = -polr's c_j. OE stores cutpoints in
*! Stata convention, so OE ologit cutpoints match these Stata fixtures directly.

clear all
set more off
set type double

import delimited "tests/r/fixtures/inputs/ordered_input.csv", clear
assert inrange(y, 0, 3)

* --- ordered logit ---
ologit y x1 x2 x3
matrix b = e(b)
matrix V = e(V)
scalar b_x1 = b[1,1]
scalar b_x2 = b[1,2]
scalar b_x3 = b[1,3]
scalar cut1 = b[1,4]
scalar cut2 = b[1,5]
scalar cut3 = b[1,6]
scalar se_x1 = sqrt(V[1,1])
scalar se_x2 = sqrt(V[2,2])
scalar se_x3 = sqrt(V[3,3])
scalar ll = e(ll)

* --- ordered logit, robust (HC1) SEs ---
ologit y x1 x2 x3, vce(robust)
matrix Vr = e(V)
scalar ser_x1 = sqrt(Vr[1,1])
scalar ser_x2 = sqrt(Vr[2,2])
scalar ser_x3 = sqrt(Vr[3,3])

* --- ordered probit ---
oprobit y x1 x2 x3
matrix bp = e(b)
matrix Vp = e(V)
scalar b_x1_p = bp[1,1]
scalar b_x2_p = bp[1,2]
scalar b_x3_p = bp[1,3]
scalar cut1_p = bp[1,4]
scalar cut2_p = bp[1,5]
scalar cut3_p = bp[1,6]
scalar se_x1_p = sqrt(Vp[1,1])
scalar se_x2_p = sqrt(Vp[2,2])
scalar se_x3_p = sqrt(Vp[3,3])
scalar ll_p = e(ll)

* --- build name/value dataset ---
clear
set obs 30
gen str32 name = ""
gen double value = .

replace name = "ologit_b_x1"  in 1
replace value = b_x1          in 1
replace name = "ologit_b_x2"  in 2
replace value = b_x2          in 2
replace name = "ologit_b_x3"  in 3
replace value = b_x3          in 3
replace name = "ologit_cut1"  in 4
replace value = cut1          in 4
replace name = "ologit_cut2"  in 5
replace value = cut2          in 5
replace name = "ologit_cut3"  in 6
replace value = cut3          in 6
replace name = "ologit_se_x1" in 7
replace value = se_x1         in 7
replace name = "ologit_se_x2" in 8
replace value = se_x2         in 8
replace name = "ologit_se_x3" in 9
replace value = se_x3         in 9
replace name = "ologit_ll"    in 10
replace value = ll            in 10
replace name = "ologit_ser_x1" in 21
replace value = ser_x1        in 21
replace name = "ologit_ser_x2" in 22
replace value = ser_x2        in 22
replace name = "ologit_ser_x3" in 23
replace value = ser_x3        in 23

replace name = "oprobit_b_x1"  in 11
replace value = b_x1_p         in 11
replace name = "oprobit_b_x2"  in 12
replace value = b_x2_p         in 12
replace name = "oprobit_b_x3"  in 13
replace value = b_x3_p         in 13
replace name = "oprobit_cut1"  in 14
replace value = cut1_p         in 14
replace name = "oprobit_cut2"  in 15
replace value = cut2_p         in 15
replace name = "oprobit_cut3"  in 16
replace value = cut3_p         in 16
replace name = "oprobit_se_x1" in 17
replace value = se_x1_p        in 17
replace name = "oprobit_se_x2" in 18
replace value = se_x2_p        in 18
replace name = "oprobit_se_x3" in 19
replace value = se_x3_p        in 19
replace name = "oprobit_ll"    in 20
replace value = ll_p           in 20

drop if name == ""
save "tests/stata/fixtures/expected/ordered.dta", replace
