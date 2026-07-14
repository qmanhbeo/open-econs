*! abond.do — All 8 Arellano-Bond flavors (collapsed + non-collapsed) against xtabond2.
*! xtabond2 3.7.2 (David Roodman, Feb 2025), Stata/MP 17.0.
*! Reference: https://github.com/droodman/xtabond2
clear all
set more off
capture ssc install xtabond2

import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv", clear
xtset entity time

postfile handle str32 name double value ///
    using "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\abond.dta", replace

*==============================================================================
* COLLAPSED FLAVOURS  (gmm(L.y, lag(2 4) collapse))
*==============================================================================

*--- Flavour 1: collapsed, one-step, non-robust ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel small
matrix a1 = e(ar1)
matrix a2 = e(ar2)
post handle ("b_Ly_c_1s_nr") (_b[L.y])
post handle ("b_x_c_1s_nr")  (_b[x])
post handle ("b_z_c_1s_nr")  (_b[z])
post handle ("se_Ly_c_1s_nr") (_se[L.y])
post handle ("se_x_c_1s_nr") (_se[x])
post handle ("se_z_c_1s_nr") (_se[z])
post handle ("N_c_1s_nr")    (e(N))
post handle ("j_c_1s_nr")    (e(j))
post handle ("N_g_c_1s_nr")  (e(N_g))
post handle ("ar1_c_1s_nr")  (a1[1,1])
post handle ("ar2_c_1s_nr")  (a2[1,1])

*--- Flavour 2: collapsed, two-step, non-robust ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel twostep small
matrix a1 = e(ar1)
matrix a2 = e(ar2)
post handle ("b_Ly_c_2s_nr") (_b[L.y])
post handle ("b_x_c_2s_nr")  (_b[x])
post handle ("b_z_c_2s_nr")  (_b[z])
post handle ("se_Ly_c_2s_nr") (_se[L.y])
post handle ("se_x_c_2s_nr") (_se[x])
post handle ("se_z_c_2s_nr") (_se[z])
post handle ("N_c_2s_nr")    (e(N))
post handle ("j_c_2s_nr")    (e(j))
post handle ("N_g_c_2s_nr")  (e(N_g))
post handle ("ar1_c_2s_nr")  (a1[1,1])
post handle ("ar2_c_2s_nr")  (a2[1,1])

*--- Flavour 3: collapsed, one-step, robust ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel robust small
matrix a1 = e(ar1)
matrix a2 = e(ar2)
post handle ("b_Ly_c_1s_r") (_b[L.y])
post handle ("b_x_c_1s_r")  (_b[x])
post handle ("b_z_c_1s_r")  (_b[z])
post handle ("se_Ly_c_1s_r") (_se[L.y])
post handle ("se_x_c_1s_r") (_se[x])
post handle ("se_z_c_1s_r") (_se[z])
post handle ("N_c_1s_r")    (e(N))
post handle ("j_c_1s_r")    (e(j))
post handle ("N_g_c_1s_r")  (e(N_g))
post handle ("ar1_c_1s_r")  (a1[1,1])
post handle ("ar2_c_1s_r")  (a2[1,1])

*--- Flavour 4: collapsed, two-step, robust ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel twostep robust small
matrix a1 = e(ar1)
matrix a2 = e(ar2)
post handle ("b_Ly_c_2s_r") (_b[L.y])
post handle ("b_x_c_2s_r")  (_b[x])
post handle ("b_z_c_2s_r")  (_b[z])
post handle ("se_Ly_c_2s_r") (_se[L.y])
post handle ("se_x_c_2s_r") (_se[x])
post handle ("se_z_c_2s_r") (_se[z])
post handle ("N_c_2s_r")    (e(N))
post handle ("j_c_2s_r")    (e(j))
post handle ("N_g_c_2s_r")  (e(N_g))
post handle ("ar1_c_2s_r")  (a1[1,1])
post handle ("ar2_c_2s_r")  (a2[1,1])

*==============================================================================
* NON-COLLAPSED FLAVOURS  (gmm(L.y, lag(2 4)))
*==============================================================================

*--- Flavour 5: non-collapsed, one-step, non-robust ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4)) iv(x z) nolevel small
matrix a1 = e(ar1)
matrix a2 = e(ar2)
post handle ("b_Ly_nc_1s_nr") (_b[L.y])
post handle ("b_x_nc_1s_nr")  (_b[x])
post handle ("b_z_nc_1s_nr")  (_b[z])
post handle ("se_Ly_nc_1s_nr") (_se[L.y])
post handle ("se_x_nc_1s_nr") (_se[x])
post handle ("se_z_nc_1s_nr") (_se[z])
post handle ("N_nc_1s_nr")    (e(N))
post handle ("j_nc_1s_nr")    (e(j))
post handle ("N_g_nc_1s_nr")  (e(N_g))
post handle ("ar1_nc_1s_nr")  (a1[1,1])
post handle ("ar2_nc_1s_nr")  (a2[1,1])

*--- Flavour 6: non-collapsed, two-step, non-robust ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4)) iv(x z) nolevel twostep small
matrix a1 = e(ar1)
matrix a2 = e(ar2)
post handle ("b_Ly_nc_2s_nr") (_b[L.y])
post handle ("b_x_nc_2s_nr")  (_b[x])
post handle ("b_z_nc_2s_nr")  (_b[z])
post handle ("se_Ly_nc_2s_nr") (_se[L.y])
post handle ("se_x_nc_2s_nr") (_se[x])
post handle ("se_z_nc_2s_nr") (_se[z])
post handle ("N_nc_2s_nr")    (e(N))
post handle ("j_nc_2s_nr")    (e(j))
post handle ("N_g_nc_2s_nr")  (e(N_g))
post handle ("ar1_nc_2s_nr")  (a1[1,1])
post handle ("ar2_nc_2s_nr")  (a2[1,1])

*--- Flavour 7: non-collapsed, one-step, robust ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4)) iv(x z) nolevel robust small
matrix a1 = e(ar1)
matrix a2 = e(ar2)
post handle ("b_Ly_nc_1s_r") (_b[L.y])
post handle ("b_x_nc_1s_r")  (_b[x])
post handle ("b_z_nc_1s_r")  (_b[z])
post handle ("se_Ly_nc_1s_r") (_se[L.y])
post handle ("se_x_nc_1s_r") (_se[x])
post handle ("se_z_nc_1s_r") (_se[z])
post handle ("N_nc_1s_r")    (e(N))
post handle ("j_nc_1s_r")    (e(j))
post handle ("N_g_nc_1s_r")  (e(N_g))
post handle ("ar1_nc_1s_r")  (a1[1,1])
post handle ("ar2_nc_1s_r")  (a2[1,1])

*--- Flavour 8: non-collapsed, two-step, robust ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4)) iv(x z) nolevel twostep robust small
matrix a1 = e(ar1)
matrix a2 = e(ar2)
post handle ("b_Ly_nc_2s_r") (_b[L.y])
post handle ("b_x_nc_2s_r")  (_b[x])
post handle ("b_z_nc_2s_r")  (_b[z])
post handle ("se_Ly_nc_2s_r") (_se[L.y])
post handle ("se_x_nc_2s_r") (_se[x])
post handle ("se_z_nc_2s_r") (_se[z])
post handle ("N_nc_2s_r")    (e(N))
post handle ("j_nc_2s_r")    (e(j))
post handle ("N_g_nc_2s_r")  (e(N_g))
post handle ("ar1_nc_2s_r")  (a1[1,1])
post handle ("ar2_nc_2s_r")  (a2[1,1])

*--- Metadata row ---
post handle ("xtabond2_version") (3.7)

postclose handle
