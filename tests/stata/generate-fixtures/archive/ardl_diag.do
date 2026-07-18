*! DIAGNOSTIC: trace Stata ardl F_pss vs statsmodels/R (3-vs-1 divergence)
*! Not a fixture generator. Dumps raw level regress coefs, r(F), test details.
clear all
set more off
set linesize 200

import delimited "tests/r/fixtures/inputs/ardl_input.csv", clear
gen t = _n
tsset t

* ---- Replicate the exact level regression ardl.ado runs (case 3 = _cons) ----
regress lrm L(1/3).lrm L(0/1).lry L(0/3).ibo L(0/2).ide
matrix b_lvl = e(b)
scalar rss_lvl = e(rss)
scalar N_lvl = e(N)
scalar dfr_lvl = e(df_r)
di "=== LEVEL REGRESS COEFS (full precision) ==="
matrix list b_lvl, format(%20.15f)
di "rss = " %20.15f rss_lvl "  N = " N_lvl "  df_r = " dfr_lvl

* ---- The PSS F: joint test of level lag terms (ardl.ado line 426) ----
* lrvars = L.lrm ; lrxvars = L.lry L.ibo L.ide (first available lag of each x)
* detvars for case 3 = _cons (unrestricted, NOT in the test)
test L.lrm L.lry L.ibo L.ide
scalar F_test = r(F)
scalar df1_test = r(df)
scalar df2_test = r(df_r)
di "=== PSS F via test L.lrm L.lry L.ibo L.ide ==="
di "F = " %20.15f F_test "  df1 = " df1_test "  df2 = " df2_test
di "chi2 form r(chi2) = " %20.15f r(chi2)

* ---- Now the actual ardl command for comparison ----
di "=== ardl ... , ec ==="
ardl lrm lry ibo ide, lags(3 1 3 2) ec
di "e(F_pss) = " %20.15f e(F_pss)
di "e(t_pss) = " %20.15f e(t_pss)
matrix bb = e(b)
matrix list bb, format(%20.15f)
