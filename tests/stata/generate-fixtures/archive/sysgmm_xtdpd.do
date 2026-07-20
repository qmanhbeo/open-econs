*! sysgmm.do — System GMM (Blundell-Bond) parity fixtures against Stata xtdpd / xtdpdsys.
*! Stata/MP 17.0.  xtdpd is base (no install).  xtdpdsys == xtdpd, level.
*!
*! IMPORTANT CONVENTION NOTES (extracted from xtdpd.ado):
*!  - xtdpd has NO `collapse` suboption and NO `collapse` main option. Its GMM
*!    instruments are inherently the collapsed-per-lag form (one instrument per
*!    lag depth) -- this corresponds to abond's "collapsed" flavor. The abond
*!    "non-collapsed" (full Arellano-Bond) instrument set has NO xtdpd equivalent.
*!  - dgmmiv(var, lag(first last)): difference-eq GMM instruments = lags [first,last]
*!    of the LEVEL of var (default lag(2 .)).  NO collapse allowed.
*!  - lgmmiv(var, lag(#)): level-eq GMM instruments = lag # of the DIFFERENCE of
*!    var (default lag(1)).  Accepts a SINGLE integer only, NOT a range.
*!  - iv(var): standard IV in BOTH equations.
*!  - e(j) / e(jdf) do NOT exist. Instrument count is e(zrank). Hansen J is NOT
*!    returned by xtdpd (only e(sargan) for non-robust; empty for robust).
*!  - AR tests are scalars e(arm1) / e(arm2) (NOT matrices e(ar1)/e(ar2) like
*!    xtabond2). They are EMPTY (.) for one-step estimations.
*!  - e(sig2) = level-error variance sigma^2 (the level-equation weight scale).
*!  - xtdpdsys in Stata 17 does NOT accept gmm()/lgmmiv()/dgmmiv(); it auto-derives
*!    default instruments (L(2/.).y diff, LD.y level). Run as plain alias only.

clear all
set more off

import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear
xtset entity time

postfile handle str32 name double value ///
    using "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\sysgmm.dta", replace

*==============================================================================
* SYSTEM GMM via xtdpd — collapsed-per-lag instrument set (xtdpd's native form).
*   dgmmiv(L.y, lag(2 4)) -> diff eq: y_{t-2},y_{t-3},y_{t-4} (levels)
*   lgmmiv(L.y, lag(1))   -> level eq: D.y_{t-1}
*   iv(x z)               -> standard IV in both eqs (+ _cons in level eq)
*==============================================================================

*--- Flavour 1: one-step, non-robust ---
xtdpd y L.y x z, dgmmiv(L.y, lag(2 4)) lgmmiv(L.y, lag(1)) iv(x z)
post handle ("b_Ly_c_1s_nr") (_b[L.y])
post handle ("b_x_c_1s_nr")  (_b[x])
post handle ("b_z_c_1s_nr")  (_b[z])
post handle ("se_Ly_c_1s_nr") (_se[L.y])
post handle ("se_x_c_1s_nr") (_se[x])
post handle ("se_z_c_1s_nr") (_se[z])
post handle ("N_c_1s_nr")    (e(N))
post handle ("j_c_1s_nr")    (e(zrank))
post handle ("N_g_c_1s_nr")  (e(N_g))
post handle ("sargan_c_1s_nr") (e(sargan))
post handle ("ar1_c_1s_nr")  (e(arm1))
post handle ("ar2_c_1s_nr")  (e(arm2))

*--- Flavour 2: two-step, non-robust ---
xtdpd y L.y x z, dgmmiv(L.y, lag(2 4)) lgmmiv(L.y, lag(1)) iv(x z) twostep
post handle ("b_Ly_c_2s_nr") (_b[L.y])
post handle ("b_x_c_2s_nr")  (_b[x])
post handle ("b_z_c_2s_nr")  (_b[z])
post handle ("se_Ly_c_2s_nr") (_se[L.y])
post handle ("se_x_c_2s_nr") (_se[x])
post handle ("se_z_c_2s_nr") (_se[z])
post handle ("N_c_2s_nr")    (e(N))
post handle ("j_c_2s_nr")    (e(zrank))
post handle ("N_g_c_2s_nr")  (e(N_g))
post handle ("sargan_c_2s_nr") (e(sargan))
post handle ("ar1_c_2s_nr")  (e(arm1))
post handle ("ar2_c_2s_nr")  (e(arm2))

*--- Flavour 3: one-step, robust ---
xtdpd y L.y x z, dgmmiv(L.y, lag(2 4)) lgmmiv(L.y, lag(1)) iv(x z) vce(robust)
post handle ("b_Ly_c_1s_r") (_b[L.y])
post handle ("b_x_c_1s_r")  (_b[x])
post handle ("b_z_c_1s_r")  (_b[z])
post handle ("se_Ly_c_1s_r") (_se[L.y])
post handle ("se_x_c_1s_r") (_se[x])
post handle ("se_z_c_1s_r") (_se[z])
post handle ("N_c_1s_r")    (e(N))
post handle ("j_c_1s_r")    (e(zrank))
post handle ("N_g_c_1s_r")  (e(N_g))
post handle ("ar1_c_1s_r")  (e(arm1))
post handle ("ar2_c_1s_r")  (e(arm2))

*--- Flavour 4: two-step, robust (WC-robust / Windmeijer-corrected) ---
xtdpd y L.y x z, dgmmiv(L.y, lag(2 4)) lgmmiv(L.y, lag(1)) iv(x z) twostep vce(robust)
post handle ("b_Ly_c_2s_r") (_b[L.y])
post handle ("b_x_c_2s_r")  (_b[x])
post handle ("b_z_c_2s_r")  (_b[z])
post handle ("se_Ly_c_2s_r") (_se[L.y])
post handle ("se_x_c_2s_r") (_se[x])
post handle ("se_z_c_2s_r") (_se[z])
post handle ("N_c_2s_r")    (e(N))
post handle ("j_c_2s_r")    (e(zrank))
post handle ("N_g_c_2s_r")  (e(N_g))
post handle ("ar1_c_2s_r")  (e(arm1))
post handle ("ar2_c_2s_r")  (e(arm2))

*==============================================================================
* xtdpdsys ALIAS (xtdpdsys == xtdpd, level ; auto-derives default instruments).
* Default instruments: diff eq L(2/.).y + LD.y D.x D.z ; level eq LD.y + _cons.
* Stata 17 xtdpdsys does NOT accept gmm()/dgmmiv()/lgmmiv().
*==============================================================================

*--- xtdpdsys, twostep robust (alias of flavour 4, different default lags) ---
xtdpdsys y L.y x z, twostep vce(robust)
post handle ("b_Ly_sys_2s_r") (_b[L.y])
post handle ("b_x_sys_2s_r")  (_b[x])
post handle ("b_z_sys_2s_r")  (_b[z])
post handle ("se_Ly_sys_2s_r") (_se[L.y])
post handle ("se_x_sys_2s_r") (_se[x])
post handle ("se_z_sys_2s_r") (_se[z])
post handle ("N_sys_2s_r")    (e(N))
post handle ("j_sys_2s_r")    (e(zrank))
post handle ("N_g_sys_2s_r")  (e(N_g))
post handle ("ar1_sys_2s_r")  (e(arm1))
post handle ("ar2_sys_2s_r")  (e(arm2))

*--- Metadata row ---
post handle ("xtdpd_version") (17)

postclose handle
