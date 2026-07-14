*! mlogit_basic.do -- Multinomial logit parity fixture (base = 1, pinned on both sides)
clear
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_mlogit.csv", clear

* --- cluster-robust estimation (also yields coefficients) ---
mlogit y x1 x2, baseoutcome(1) vce(cluster cluster)
scalar s_N        = e(N)
scalar b_2_cons   = _b[2:_cons]
scalar b_2_x1     = _b[2:x1]
scalar b_2_x2     = _b[2:x2]
scalar b_3_cons   = _b[3:_cons]
scalar b_3_x1     = _b[3:x1]
scalar b_3_x2     = _b[3:x2]
scalar se_clu_2_cons = _se[2:_cons]
scalar se_clu_2_x1   = _se[2:x1]
scalar se_clu_2_x2   = _se[2:x2]
scalar se_clu_3_cons = _se[3:_cons]
scalar se_clu_3_x1   = _se[3:x1]
scalar se_clu_3_x2   = _se[3:x2]

* --- robust estimation ---
mlogit y x1 x2, baseoutcome(1) vce(robust)
scalar se_rob_2_cons = _se[2:_cons]
scalar se_rob_2_x1   = _se[2:x1]
scalar se_rob_2_x2   = _se[2:x2]
scalar se_rob_3_cons = _se[3:_cons]
scalar se_rob_3_x1   = _se[3:x1]
scalar se_rob_3_x2   = _se[3:x2]

* --- margins (point estimates are vce-invariant); capture r(b)/r(V) ---
margins, dydx(*) predict(outcome(1))
matrix m_b = r(b)
matrix m_V = r(V)
scalar me_1_x1 = m_b[1,1]
scalar me_1_x2 = m_b[1,2]
scalar me_se_1_x1 = sqrt(m_V[1,1])
scalar me_se_1_x2 = sqrt(m_V[2,2])
margins, dydx(*) predict(outcome(2))
matrix m_b = r(b)
matrix m_V = r(V)
scalar me_2_x1 = m_b[1,1]
scalar me_2_x2 = m_b[1,2]
scalar me_se_2_x1 = sqrt(m_V[1,1])
scalar me_se_2_x2 = sqrt(m_V[2,2])
margins, dydx(*) predict(outcome(3))
matrix m_b = r(b)
matrix m_V = r(V)
scalar me_3_x1 = m_b[1,1]
scalar me_3_x2 = m_b[1,2]
scalar me_se_3_x1 = sqrt(m_V[1,1])
scalar me_se_3_x2 = sqrt(m_V[2,2])

* --- write name/value dataset ---
clear
set obs 31
gen str20 name  = ""
gen double value = .
replace name = "N"            in 1
replace name = "b_2_cons"    in 2
replace name = "b_2_x1"      in 3
replace name = "b_2_x2"      in 4
replace name = "b_3_cons"    in 5
replace name = "b_3_x1"      in 6
replace name = "b_3_x2"      in 7
replace name = "se_rob_2_cons" in 8
replace name = "se_rob_2_x1"   in 9
replace name = "se_rob_2_x2"   in 10
replace name = "se_rob_3_cons" in 11
replace name = "se_rob_3_x1"   in 12
replace name = "se_rob_3_x2"   in 13
replace name = "se_clu_2_cons" in 14
replace name = "se_clu_2_x1"   in 15
replace name = "se_clu_2_x2"   in 16
replace name = "se_clu_3_cons" in 17
replace name = "se_clu_3_x1"   in 18
replace name = "se_clu_3_x2"   in 19
replace name = "me_1_x1"      in 20
replace name = "me_1_x2"      in 21
replace name = "me_2_x1"      in 22
replace name = "me_2_x2"      in 23
replace name = "me_3_x1"      in 24
replace name = "me_3_x2"      in 25
replace name = "me_se_1_x1"   in 26
replace name = "me_se_1_x2"   in 27
replace name = "me_se_2_x1"   in 28
replace name = "me_se_2_x2"   in 29
replace name = "me_se_3_x1"   in 30
replace name = "me_se_3_x2"   in 31

replace value = s_N            in 1
replace value = b_2_cons       in 2
replace value = b_2_x1         in 3
replace value = b_2_x2         in 4
replace value = b_3_cons       in 5
replace value = b_3_x1         in 6
replace value = b_3_x2         in 7
replace value = se_rob_2_cons  in 8
replace value = se_rob_2_x1    in 9
replace value = se_rob_2_x2    in 10
replace value = se_rob_3_cons  in 11
replace value = se_rob_3_x1    in 12
replace value = se_rob_3_x2    in 13
replace value = se_clu_2_cons  in 14
replace value = se_clu_2_x1    in 15
replace value = se_clu_2_x2    in 16
replace value = se_clu_3_cons  in 17
replace value = se_clu_3_x1    in 18
replace value = se_clu_3_x2    in 19
replace value = me_1_x1        in 20
replace value = me_1_x2        in 21
replace value = me_2_x1        in 22
replace value = me_2_x2        in 23
replace value = me_3_x1        in 24
replace value = me_3_x2        in 25
replace value = me_se_1_x1     in 26
replace value = me_se_1_x2     in 27
replace value = me_se_2_x1     in 28
replace value = me_se_2_x2     in 29
replace value = me_se_3_x1     in 30
replace value = me_se_3_x2     in 31

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\mlogit_basic.dta", replace
