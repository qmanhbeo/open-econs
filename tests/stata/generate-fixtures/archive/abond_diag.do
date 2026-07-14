*! abond_diag.do - Stripped 1-step Difference GMM diagnostic
clear all
set more off
capture ssc install xtabond2
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv", clear
xtset entity time

*--- RUN A: Uncollapsed, lag(2 2), 1-step, nolevel, small ---
di as text "=== DIAGNOSTIC RUN A: Uncollapsed 1-step ==="
xtabond2 y L.y x z, gmm(L.y, lag(2 2)) iv(x z) nolevel small

scalar s_N_A      = e(N)
scalar s_ninstr_A = e(j)
scalar s_bly_A    = _b[L.y]
scalar s_bx_A     = _b[x]
scalar s_bz_A     = _b[z]
scalar s_sely_A   = _se[L.y]
scalar s_sex_A    = _se[x]
scalar s_sez_A    = _se[z]

di as text "N = " s_N_A
di as text "Number of instruments = " s_ninstr_A
di as text "b_Ly = " s_bly_A
di as text "b_x  = " s_bx_A
di as text "b_z  = " s_bz_A
di as text "se_Ly = " s_sely_A
di as text "se_x  = " s_sex_A
di as text "se_z  = " s_sez_A

matrix V_A = e(V)
di as text "--- VCV matrix (Run A) ---"
matrix list V_A

*--- RUN B: Uncollapsed, lag(2 4), 1-step, nolevel, small ---
di as text ""
di as text "=== DIAGNOSTIC RUN B: Uncollapsed full lags 1-step ==="
xtabond2 y L.y x z, gmm(L.y, lag(2 4)) iv(x z) nolevel small

scalar s_N_B      = e(N)
scalar s_ninstr_B = e(j)
scalar s_bly_B    = _b[L.y]
scalar s_bx_B     = _b[x]
scalar s_bz_B     = _b[z]
scalar s_sely_B   = _se[L.y]
scalar s_sex_B    = _se[x]
scalar s_sez_B    = _se[z]

di as text "N = " s_N_B
di as text "Number of instruments = " s_ninstr_B
di as text "b_Ly = " s_bly_B
di as text "b_x  = " s_bx_B
di as text "b_z  = " s_bz_B
di as text "se_Ly = " s_sely_B
di as text "se_x  = " s_sex_B
di as text "se_z  = " s_sez_B

matrix V_B = e(V)
di as text "--- VCV matrix (Run B) ---"
matrix list V_B

*--- RUN C: Uncollapsed, lag(2 4), 2-step, nolevel, small ---
di as text ""
di as text "=== DIAGNOSTIC RUN C: Uncollapsed full lags 2-step ==="
xtabond2 y L.y x z, gmm(L.y, lag(2 4)) iv(x z) nolevel twostep small

scalar s_N_C      = e(N)
scalar s_ninstr_C = e(j)
scalar s_bly_C    = _b[L.y]
scalar s_bx_C     = _b[x]
scalar s_bz_C     = _b[z]
scalar s_sely_C   = _se[L.y]
scalar s_sex_C    = _se[x]
scalar s_sez_C    = _se[z]

di as text "N = " s_N_C
di as text "Number of instruments = " s_ninstr_C
di as text "b_Ly = " s_bly_C
di as text "b_x  = " s_bx_C
di as text "b_z  = " s_bz_C
di as text "se_Ly = " s_sely_C
di as text "se_x  = " s_sex_C
di as text "se_z  = " s_sez_C

matrix V_C = e(V)
di as text "--- VCV matrix (Run C) ---"
matrix list V_C

*--- Save results ---
clear
set obs 21
gen str20 name  = ""
gen double value = .

replace name = "N_A"      in 1
replace name = "ninstr_A"  in 2
replace name = "b_Ly_A"   in 3
replace name = "b_x_A"    in 4
replace name = "b_z_A"    in 5
replace name = "se_Ly_A"  in 6
replace name = "se_x_A"   in 7
replace name = "se_z_A"   in 8
replace name = "N_B"      in 9
replace name = "ninstr_B"  in 10
replace name = "b_Ly_B"   in 11
replace name = "b_x_B"    in 12
replace name = "b_z_B"    in 13
replace name = "se_Ly_B"  in 14
replace name = "se_x_B"   in 15
replace name = "se_z_B"   in 16
replace name = "ninstr_C"  in 17
replace name = "b_Ly_C"   in 18
replace name = "se_Ly_C"  in 19
replace name = "se_x_C"   in 20

replace value = s_N_A      in 1
replace value = s_ninstr_A  in 2
replace value = s_bly_A    in 3
replace value = s_bx_A     in 4
replace value = s_bz_A     in 5
replace value = s_sely_A   in 6
replace value = s_sex_A    in 7
replace value = s_sez_A    in 8
replace value = s_N_B      in 9
replace value = s_ninstr_B  in 10
replace value = s_bly_B    in 11
replace value = s_bx_B     in 12
replace value = s_bz_B     in 13
replace value = s_sely_B   in 14
replace value = s_sex_B    in 15
replace value = s_sez_B    in 16
replace value = s_ninstr_C  in 17
replace value = s_bly_C    in 18
replace value = s_sely_C   in 19
replace value = s_sex_C    in 20

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\abond_diag.dta", replace
