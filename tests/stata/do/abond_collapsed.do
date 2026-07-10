*! abond_collapsed.do - Collapsed Difference GMM diagnostic
clear all
set more off
capture ssc install xtabond2
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv", clear
xtset entity time

*--- Try collapse as sub-option of gmm() ---
di as text "=== COLLAPSED Run B: lag(2 4) 1-step (collapse in gmm) ==="
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel small

scalar s_N_B      = e(N)
scalar s_ninstr_B = e(j)
scalar s_bly_B    = _b[L.y]
scalar s_bx_B     = _b[x]
scalar s_bz_B     = _b[z]
scalar s_sely_B   = _se[L.y]
scalar s_sex_B    = _se[x]
scalar s_sez_B    = _se[z]

matrix V_B = e(V)
di as text "b_Ly=" s_bly_B " b_x=" s_bx_B " b_z=" s_bz_B
di as text "se_Ly=" s_sely_B " se_x=" s_sex_B " se_z=" s_sez_B
di as text "ninstr=" s_ninstr_B
di as text "VCV:"
matrix list V_B

*--- COLLAPSED Run C: lag(2 4), 2-step, nolevel, small ---
di as text ""
di as text "=== COLLAPSED Run C: lag(2 4) 2-step ==="
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel twostep small

scalar s_N_C      = e(N)
scalar s_ninstr_C = e(j)
scalar s_bly_C    = _b[L.y]
scalar s_bx_C     = _b[x]
scalar s_bz_C     = _b[z]
scalar s_sely_C   = _se[L.y]
scalar s_sex_C    = _se[x]
scalar s_sez_C    = _se[z]

matrix V_C = e(V)
di as text "b_Ly=" s_bly_C " b_x=" s_bx_C " b_z=" s_bz_C
di as text "se_Ly=" s_sely_C " se_x=" s_sex_C " se_z=" s_sez_C
di as text "ninstr=" s_ninstr_C
di as text "VCV:"
matrix list V_C
