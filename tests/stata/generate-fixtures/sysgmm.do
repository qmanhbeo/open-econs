*! sysgmm.do â€” System GMM (Blundell-Bond) parity fixtures against Stata xtabond2.
*! xtabond2 3.7.2 (David Roodman, Feb 2025), Stata/MP 17.0.
*! Reference: https://github.com/droodman/xtabond2
*!
*! RE-ANCHORING NOTE (2026-07-19): the previous sysgmm fixture was generated with
*! xtdpd and is WRONG for parity against xtabond2. xtdpd's difference-equation
*! GMM-instrument convention differs from xtabond2 (xtabond2 is the target because
*! open_econs abond() already matches xtabond2 difference GMM to 1e-6). We now
*! regenerate with xtabond2 system-GMM so that the diff-eq part is identical to
*! abond()'s collapsed flavors and the level-eq is stacked exactly as xtabond2 does.
*!
*! CANONICAL SYSTEM-GMM CALL (matches Blundell-Bond, explicit eq() suboptions):
*!   xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
*!                         gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) ///
*!                         twostep robust
*!
*! CONVENTION NOTES (extracted from xtabond2.ado / .hlp):
*!  - eq(diff): instruments enter the DIFFERENCE equation.
*!  - eq(level): instruments enter the LEVEL equation (system GMM).
*!  - collapse: collapse the GMM instruments to one per lag depth (matches abond
*!    collapsed flavor). Non-collapsed system GMM is not our target.
*!  - noleveleq is NOT used -> both equations estimated (system GMM).
*!  - AR scalars are e(ar1)/e(ar2) (matrices, read [1,1]); EMPTY (.) for one-step.
*!  - Hansen J: e(hansen) (stat) / e(hansenp) (p); Sargan: e(sargan)/e(sarganp).
*!    These exist only when the corresponding moment condition is available
*!    (Sargan for one-step/non-robust; Hansen for two-step; robust blocks both
*!    Sargan in some settings). Captured as-is; NaN if empty.
*!  - Instrument count: e(rank) is EMPTY for system GMM. Use e(j0) = total
*!    moment conditions (= number of instruments). For this controlled set
*!    e(j0)==11 (3 diff-GMM + D.x + D.z + 1 level-GMM + x + z + _cons). e(j) is
*!    the J-stat df (e(j0) - #regressors). xtdpdsys auto set gives e(zrank)==13.
*!  - e(sig2) IS returned by xtabond2 (level-error variance). Captured as sig2.

clear all
set more off
capture ssc install xtabond2

set type double
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear
xtset entity time

postfile handle str32 name double value ///
    using "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\sysgmm.dta", replace

*==============================================================================
* SYSTEM GMM via xtabond2 (Blundell-Bond).
*   diff eq : gmm(L.y, lag(2 4) collapse) + iv(x z, eq(diff))
*   level eq: gmm(L.y, lag(1)   collapse) + iv(x z, eq(level))  (no _cons in diff eq;
*             _cons enters the level eq as the regressor constant)
*   Total instruments: 3 (diff GMM collapsed) + D.x + D.z + 1 (level GMM collapsed)
*                       + x + z + _cons = 8  -> e(rank) == 8.
*==============================================================================

*--- Flavour 1: one-step, non-robust ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                       gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) small
matrix a1 = e(ar1)
matrix a2 = e(ar2)
post handle ("b_Ly_c_1s_nr") (_b[L.y])
post handle ("b_x_c_1s_nr")  (_b[x])
post handle ("b_z_c_1s_nr")  (_b[z])
post handle ("b_cons_c_1s_nr") (_b[_cons])
post handle ("se_Ly_c_1s_nr") (_se[L.y])
post handle ("se_x_c_1s_nr")  (_se[x])
post handle ("se_z_c_1s_nr")  (_se[z])
post handle ("se_cons_c_1s_nr") (_se[_cons])
post handle ("N_c_1s_nr")    (e(N))
post handle ("zrank_c_1s_nr") (e(j0))
post handle ("N_g_c_1s_nr")  (e(N_g))
post handle ("ar1_c_1s_nr")  (a1[1,1])
post handle ("ar2_c_1s_nr")  (a2[1,1])
post handle ("sargan_p_c_1s_nr") (e(sarganp))
post handle ("hansen_j_c_1s_nr") (e(hansen))
post handle ("hansen_p_c_1s_nr") (e(hansenp))
post handle ("sig2_c_1s_nr") (e(sig2))

*--- Flavour 2: two-step, non-robust ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                       gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) twostep small
matrix a1 = e(ar1)
matrix a2 = e(ar2)
post handle ("b_Ly_c_2s_nr") (_b[L.y])
post handle ("b_x_c_2s_nr")  (_b[x])
post handle ("b_z_c_2s_nr")  (_b[z])
post handle ("b_cons_c_2s_nr") (_b[_cons])
post handle ("se_Ly_c_2s_nr") (_se[L.y])
post handle ("se_x_c_2s_nr")  (_se[x])
post handle ("se_z_c_2s_nr")  (_se[z])
post handle ("se_cons_c_2s_nr") (_se[_cons])
post handle ("N_c_2s_nr")    (e(N))
post handle ("zrank_c_2s_nr") (e(j0))
post handle ("N_g_c_2s_nr")  (e(N_g))
post handle ("ar1_c_2s_nr")  (a1[1,1])
post handle ("ar2_c_2s_nr")  (a2[1,1])
post handle ("sargan_p_c_2s_nr") (e(sarganp))
post handle ("hansen_j_c_2s_nr") (e(hansen))
post handle ("hansen_p_c_2s_nr") (e(hansenp))
post handle ("sig2_c_2s_nr") (e(sig2))

*--- Flavour 3: two-step, robust (Windmeijer-corrected) ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                       gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) twostep robust small
matrix a1 = e(ar1)
matrix a2 = e(ar2)
post handle ("b_Ly_c_2s_r") (_b[L.y])
post handle ("b_x_c_2s_r")  (_b[x])
post handle ("b_z_c_2s_r")  (_b[z])
post handle ("b_cons_c_2s_r") (_b[_cons])
post handle ("se_Ly_c_2s_r") (_se[L.y])
post handle ("se_x_c_2s_r")  (_se[x])
post handle ("se_z_c_2s_r")  (_se[z])
post handle ("se_cons_c_2s_r") (_se[_cons])
post handle ("N_c_2s_r")    (e(N))
post handle ("zrank_c_2s_r") (e(j0))
post handle ("N_g_c_2s_r")  (e(N_g))
post handle ("ar1_c_2s_r")  (a1[1,1])
post handle ("ar2_c_2s_r")  (a2[1,1])
post handle ("hansen_j_c_2s_r") (e(hansen))
post handle ("hansen_p_c_2s_r") (e(hansenp))
post handle ("sargan_p_c_2s_r") (e(sarganp))
post handle ("sig2_c_2s_r") (e(sig2))

*--- Flavour 4: one-step, robust (AR empty for one-step) ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                       gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) robust small
matrix a1 = e(ar1)
matrix a2 = e(ar2)
post handle ("b_Ly_c_1s_r") (_b[L.y])
post handle ("b_x_c_1s_r")  (_b[x])
post handle ("b_z_c_1s_r")  (_b[z])
post handle ("b_cons_c_1s_r") (_b[_cons])
post handle ("se_Ly_c_1s_r") (_se[L.y])
post handle ("se_x_c_1s_r")  (_se[x])
post handle ("se_z_c_1s_r")  (_se[z])
post handle ("se_cons_c_1s_r") (_se[_cons])
post handle ("N_c_1s_r")    (e(N))
post handle ("zrank_c_1s_r") (e(j0))
post handle ("N_g_c_1s_r")  (e(N_g))
post handle ("ar1_c_1s_r")  (a1[1,1])
post handle ("ar2_c_1s_r")  (a2[1,1])
post handle ("hansen_j_c_1s_r") (e(hansen))
post handle ("hansen_p_c_1s_r") (e(hansenp))
post handle ("sargan_p_c_1s_r") (e(sarganp))
post handle ("sig2_c_1s_r") (e(sig2))

*==============================================================================
* DOCUMENTATION ONLY: xtdpdsys alias (NOT the parity target).
* xtdpdsys auto-derives default instrument set: diff eq L(2/.).y + D.x + D.z ;
* level eq LD.y + x + z + _cons  -> 13 instruments (different from controlled 8).
* Record numbers for confirmation that it is a DIFFERENT (larger) set.
*==============================================================================
xtdpdsys y L.y x z, twostep vce(robust)
matrix a1 = e(arm1)
matrix a2 = e(arm2)
post handle ("b_Ly_sys_2s_r") (_b[L.y])
post handle ("b_x_sys_2s_r")  (_b[x])
post handle ("b_z_sys_2s_r")  (_b[z])
post handle ("se_Ly_sys_2s_r") (_se[L.y])
post handle ("se_x_sys_2s_r")  (_se[x])
post handle ("se_z_sys_2s_r")  (_se[z])
post handle ("N_sys_2s_r")    (e(N))
post handle ("zrank_sys_2s_r")  (e(zrank))
post handle ("N_g_sys_2s_r")  (e(N_g))
post handle ("ar1_sys_2s_r")  (a1[1,1])
post handle ("ar2_sys_2s_r")  (a2[1,1])

*--- Metadata ---
post handle ("xtabond2_version") (3.7)
post handle ("model") (1)   // 1 = system GMM via xtabond2

postclose handle


