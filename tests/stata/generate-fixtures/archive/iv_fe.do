*! iv_fe.do — IV/2SLS with entity fixed effects, Stata xtivreg,fe parity
*! Captures robust, nonrobust, and HC1 (vce(robust) equivalent) SEs.
*! xtivreg,fe sweeps the intercept, so only w and x are reported.
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_iv_panel.csv", clear
xtset id t

* robust
xtivreg y w (x = z1 z2), fe vce(robust)
scalar s_bw_r  = _b[w]
scalar s_bx_r  = _b[x]
scalar s_sew_r = _se[w]
scalar s_sex_r = _se[x]

* nonrobust (iid)
xtivreg y w (x = z1 z2), fe
scalar s_bw_n  = _b[w]
scalar s_bx_n  = _b[x]
scalar s_sew_n = _se[w]
scalar s_sex_n = _se[x]

* N, N_g, df_r for documentation
scalar s_N   = e(N)
scalar s_Ng  = e(N_g)
scalar s_dfr = e(df_rz)

clear
set obs 14
gen str20 name = ""
gen double value = .
replace name = "bw_r" in 1
replace value = s_bw_r in 1
replace name = "bx_r" in 2
replace value = s_bx_r in 2
replace name = "sew_r" in 3
replace value = s_sew_r in 3
replace name = "sex_r" in 4
replace value = s_sex_r in 4
replace name = "bw_n" in 5
replace value = s_bw_n in 5
replace name = "bx_n" in 6
replace value = s_bx_n in 6
replace name = "sew_n" in 7
replace value = s_sew_n in 7
replace name = "sex_n" in 8
replace value = s_sex_n in 8
replace name = "N" in 9
replace value = s_N in 9
replace name = "Ng" in 10
replace value = s_Ng in 10
replace name = "dfr" in 11
replace value = s_dfr in 11

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\iv_fe.dta", replace
