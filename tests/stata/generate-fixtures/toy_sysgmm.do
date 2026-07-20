*! toy_sysgmm.do â€” Tiny synthetic panel for system-GMM AR test triangulation.
clear all
set more off

set type double
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\toy_sysgmm.csv", clear
xtset entity time

postfile handle str32 name double value ///
    using "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\toy_sysgmm.dta", replace

*==============================================================================
* SYSTEM GMM via xtabond2 on 3-entity T=4 toy panel
*==============================================================================

*--- Flavour 1: one-step, non-robust ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                       gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) small
matrix a1 = e(ar1)
matrix a2 = e(ar2)
post handle ("b_Ly_1s_nr") (_b[L.y])
post handle ("b_x_1s_nr")  (_b[x])
post handle ("b_z_1s_nr")  (_b[z])
post handle ("b_cons_1s_nr") (_b[_cons])
post handle ("se_Ly_1s_nr") (_se[L.y])
post handle ("se_x_1s_nr")  (_se[x])
post handle ("se_z_1s_nr")  (_se[z])
post handle ("se_cons_1s_nr") (_se[_cons])
post handle ("N_1s_nr")    (e(N))
post handle ("zrank_1s_nr") (e(j0))
post handle ("N_g_1s_nr")  (e(N_g))
post handle ("ar1_1s_nr")  (a1[1,1])
post handle ("ar2_1s_nr")  (a2[1,1])
post handle ("sig2_1s_nr") (e(sig2))

*--- Flavour 2: two-step, non-robust ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                       gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) twostep small
matrix a1 = e(ar1)
matrix a2 = e(ar2)
post handle ("b_Ly_2s_nr") (_b[L.y])
post handle ("b_x_2s_nr")  (_b[x])
post handle ("b_z_2s_nr")  (_b[z])
post handle ("b_cons_2s_nr") (_b[_cons])
post handle ("se_Ly_2s_nr") (_se[L.y])
post handle ("se_x_2s_nr")  (_se[x])
post handle ("se_z_2s_nr")  (_se[z])
post handle ("se_cons_2s_nr") (_se[_cons])
post handle ("N_2s_nr")    (e(N))
post handle ("zrank_2s_nr") (e(j0))
post handle ("N_g_2s_nr")  (e(N_g))
post handle ("ar1_2s_nr")  (a1[1,1])
post handle ("ar2_2s_nr")  (a2[1,1])
post handle ("sig2_2s_nr") (e(sig2))

*--- Flavour 3: two-step, robust ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                       gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) twostep robust small
matrix a1 = e(ar1)
matrix a2 = e(ar2)
post handle ("b_Ly_2s_r") (_b[L.y])
post handle ("b_x_2s_r")  (_b[x])
post handle ("b_z_2s_r")  (_b[z])
post handle ("b_cons_2s_r") (_b[_cons])
post handle ("se_Ly_2s_r") (_se[L.y])
post handle ("se_x_2s_r")  (_se[x])
post handle ("se_z_2s_r")  (_se[z])
post handle ("se_cons_2s_r") (_se[_cons])
post handle ("N_2s_r")    (e(N))
post handle ("zrank_2s_r") (e(j0))
post handle ("N_g_2s_r")  (e(N_g))
post handle ("ar1_2s_r")  (a1[1,1])
post handle ("ar2_2s_r")  (a2[1,1])
post handle ("sig2_2s_r") (e(sig2))

*--- Flavour 4: one-step, robust ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                       gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) robust small
matrix a1 = e(ar1)
matrix a2 = e(ar2)
post handle ("b_Ly_1s_r") (_b[L.y])
post handle ("b_x_1s_r")  (_b[x])
post handle ("b_z_1s_r")  (_b[z])
post handle ("b_cons_1s_r") (_b[_cons])
post handle ("se_Ly_1s_r") (_se[L.y])
post handle ("se_x_1s_r")  (_se[x])
post handle ("se_z_1s_r")  (_se[z])
post handle ("se_cons_1s_r") (_se[_cons])
post handle ("N_1s_r")    (e(N))
post handle ("zrank_1s_r") (e(j0))
post handle ("N_g_1s_r")  (e(N_g))
post handle ("ar1_1s_r")  (a1[1,1])
post handle ("ar2_1s_r")  (a2[1,1])
post handle ("sig2_1s_r") (e(sig2))

postclose handle
