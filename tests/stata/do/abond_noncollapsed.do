*! abond_noncollapsed.do — Non-collapsed Difference GMM ground truth (all 4 flavors)
clear all
set more off
capture ssc install xtabond2

import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv", clear
xtset entity time

*--- Flavor 1: one-step, non-robust ---
di as text "=== 1: one-step, non-robust ==="
xtabond2 y L.y x z, gmm(L.y, lag(2 4)) iv(x z) nolevel small

scalar f1_N      = e(N)
scalar f1_ninstr = e(j)
scalar f1_ne     = e(N_g)
scalar f1_bLy    = _b[L.y]
scalar f1_bx     = _b[x]
scalar f1_bz     = _b[z]
scalar f1_seLy   = _se[L.y]
scalar f1_sex    = _se[x]
scalar f1_sez    = _se[z]
matrix a1 = e(ar1)
scalar f1_ar1z   = a1[1,1]
matrix a2 = e(ar2)
scalar f1_ar2z   = a2[1,1]

di "bLy=" f1_bLy " bx=" f1_bx " bz=" f1_bz
di "seLy=" f1_seLy " sex=" f1_sex " sez=" f1_sez
di "ar1z=" f1_ar1z " ar2z=" f1_ar2z

*--- Flavor 2: two-step, non-robust ---
di as text "=== 2: two-step, non-robust ==="
xtabond2 y L.y x z, gmm(L.y, lag(2 4)) iv(x z) nolevel twostep small

scalar f2_bLy    = _b[L.y]
scalar f2_bx     = _b[x]
scalar f2_bz     = _b[z]
scalar f2_seLy   = _se[L.y]
scalar f2_sex    = _se[x]
scalar f2_sez    = _se[z]
matrix a1_2 = e(ar1)
scalar f2_ar1z   = a1_2[1,1]
matrix a2_2 = e(ar2)
scalar f2_ar2z   = a2_2[1,1]

di "bLy=" f2_bLy " bx=" f2_bx " bz=" f2_bz
di "seLy=" f2_seLy " sex=" f2_sex " sez=" f2_sez
di "ar1z=" f2_ar1z " ar2z=" f2_ar2z

*--- Flavor 3: one-step, robust ---
di as text "=== 3: one-step, robust ==="
xtabond2 y L.y x z, gmm(L.y, lag(2 4)) iv(x z) nolevel robust small

scalar f3_bLy    = _b[L.y]
scalar f3_bx     = _b[x]
scalar f3_bz     = _b[z]
scalar f3_seLy   = _se[L.y]
scalar f3_sex    = _se[x]
scalar f3_sez    = _se[z]
matrix a1_3 = e(ar1)
scalar f3_ar1z   = a1_3[1,1]
matrix a2_3 = e(ar2)
scalar f3_ar2z   = a2_3[1,1]

di "bLy=" f3_bLy " bx=" f3_bx " bz=" f3_bz
di "seLy=" f3_seLy " sex=" f3_sex " sez=" f3_sez
di "ar1z=" f3_ar1z " ar2z=" f3_ar2z

*--- Flavor 4: two-step, robust ---
di as text "=== 4: two-step, robust ==="
xtabond2 y L.y x z, gmm(L.y, lag(2 4)) iv(x z) nolevel twostep robust small

scalar f4_bLy    = _b[L.y]
scalar f4_bx     = _b[x]
scalar f4_bz     = _b[z]
scalar f4_seLy   = _se[L.y]
scalar f4_sex    = _se[x]
scalar f4_sez    = _se[z]
matrix a1_4 = e(ar1)
scalar f4_ar1z   = a1_4[1,1]
matrix a2_4 = e(ar2)
scalar f4_ar2z   = a2_4[1,1]

di "bLy=" f4_bLy " bx=" f4_bx " bz=" f4_bz
di "seLy=" f4_seLy " sex=" f4_sex " sez=" f4_sez
di "ar1z=" f4_ar1z " ar2z=" f4_ar2z

*--- Save to .dta (explicit, no foreach) ---
clear
set obs 48
gen str20 name  = ""
gen double value = .

replace name = "N"     in 1
replace name = "ninstr" in 2
replace name = "ne"    in 3
replace name = "bLy"   in 4
replace name = "bx"    in 5
replace name = "bz"    in 6
replace name = "seLy"  in 7
replace name = "sex"   in 8
replace name = "sez"   in 9
replace name = "ar1z"  in 10
replace name = "ar2z"  in 11

replace name = "bLy2"  in 12
replace name = "bx2"   in 13
replace name = "bz2"   in 14
replace name = "seLy2" in 15
replace name = "sex2"  in 16
replace name = "sez2"  in 17
replace name = "ar1z2" in 18
replace name = "ar2z2" in 19

replace name = "bLy3"  in 20
replace name = "bx3"   in 21
replace name = "bz3"   in 22
replace name = "seLy3" in 23
replace name = "sex3"  in 24
replace name = "sez3"  in 25
replace name = "ar1z3" in 26
replace name = "ar2z3" in 27

replace name = "bLy4"  in 28
replace name = "bx4"   in 29
replace name = "bz4"   in 30
replace name = "seLy4" in 31
replace name = "sex4"  in 32
replace name = "sez4"  in 33
replace name = "ar1z4" in 34
replace name = "ar2z4" in 35

replace name = "ninstr2" in 36
replace name = "ninstr3" in 37
replace name = "ninstr4" in 38
replace name = "N2" in 39
replace name = "N3" in 40
replace name = "N4" in 41
replace name = "ne2" in 42
replace name = "ne3" in 43
replace name = "ne4" in 44

* flavor 1
replace value = f1_N      in 1
replace value = f1_ninstr in 2
replace value = f1_ne     in 3
replace value = f1_bLy    in 4
replace value = f1_bx     in 5
replace value = f1_bz     in 6
replace value = f1_seLy   in 7
replace value = f1_sex    in 8
replace value = f1_sez    in 9
replace value = f1_ar1z   in 10
replace value = f1_ar2z   in 11

* flavor 2
replace value = f2_bLy    in 12
replace value = f2_bx     in 13
replace value = f2_bz     in 14
replace value = f2_seLy   in 15
replace value = f2_sex    in 16
replace value = f2_sez    in 17
replace value = f2_ar1z   in 18
replace value = f2_ar2z   in 19

* flavor 3
replace value = f3_bLy    in 20
replace value = f3_bx     in 21
replace value = f3_bz     in 22
replace value = f3_seLy   in 23
replace value = f3_sex    in 24
replace value = f3_sez    in 25
replace value = f3_ar1z   in 26
replace value = f3_ar2z   in 27

* flavor 4
replace value = f4_bLy    in 28
replace value = f4_bx     in 29
replace value = f4_bz     in 30
replace value = f4_seLy   in 31
replace value = f4_sex    in 32
replace value = f4_sez    in 33
replace value = f4_ar1z   in 34
replace value = f4_ar2z   in 35

* misc
replace value = f1_ninstr in 36
replace value = f1_ninstr in 37
replace value = f1_ninstr in 38
replace value = f1_N      in 39
replace value = f1_N      in 40
replace value = f1_N      in 41
replace value = f1_ne     in 42
replace value = f1_ne     in 43
replace value = f1_ne     in 44

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\abond_noncollapsed.dta", replace
