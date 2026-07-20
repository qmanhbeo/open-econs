* Probe Stata xtabond2 to understand small-sample multiplier formula
clear all
set more off

* Check if xtabond2 is installed
capture which xtabond2
if _rc != 0 {
    ssc install xtabond2, replace
}
which xtabond2

* Find the ado file path
findfile xtabond2.ado
display "ADO path: " `"`r(fn)'"'

* Read the relevant formulas
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear
xtset entity time

* Compute for diff-only, two-step, non-robust, small
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel twostep small
display "e(N) = " e(N)
display "e(N_g) = " e(N_g)
matrix list e(V)

* Compute the small multiplier by comparing raw V with small-corrected V
* First, estimate without small to get the base V
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel twostep
display "Without small:"
matrix V_nosmall = e(V)
matrix list V_nosmall

* Now with small
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel twostep small
display "With small:"
matrix V_small = e(V)
matrix list V_small

* Compute ratio
forvalues i = 1 / `=colsof(V_nosmall)' {
    forvalues j = 1 / `=rowsof(V_nosmall)' {
        if V_nosmall[`i',`j'] != 0 {
            display "Ratio V[`i',`j'] = " V_small[`i',`j'] / V_nosmall[`i',`j']
        }
    }
}

* Now for one-step robust (non-onestepnonrobust)
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel robust small
matrix V_robust_small = e(V)
matrix list V_robust_small

xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel robust
matrix V_robust_nosmall = e(V)
matrix list V_robust_nosmall

forvalues i = 1 / `=colsof(V_robust_small)' {
    forvalues j = 1 / `=rowsof(V_robust_small)' {
        if V_robust_nosmall[`i',`j'] != 0 {
            display "Ratio V_robust[`i',`j'] = " V_robust_small[`i',`j'] / V_robust_nosmall[`i',`j']
        }
    }
}

* System GMM probe
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                   gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) twostep small
matrix V_sys_small = e(V)
matrix list V_sys_small
display "e(N) = " e(N)
display "e(N_g) = " e(N_g)

xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                   gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) twostep
matrix V_sys_nosmall = e(V)
matrix list V_sys_nosmall

forvalues i = 1 / `=colsof(V_sys_small)' {
    forvalues j = 1 / `=rowsof(V_sys_small)' {
        if V_sys_nosmall[`i',`j'] != 0 {
            display "Ratio V_sys[`i',`j'] = " V_sys_small[`i',`j'] / V_sys_nosmall[`i',`j']
        }
    }
}
