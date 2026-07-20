*! probe_ar_deep.do — Extract per-entity values used in _ARTests for system GMM
clear all
set more off
set matsize 5000

import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear
xtset entity time

* Two-step non-robust
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                       gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) twostep small

* Get AR test scalars
matrix a1 = e(ar1)
matrix a2 = e(ar2)
display "AR1 = " a1[1,1]
display "AR2 = " a2[1,1]

* Get sigma
display "sig2 = " e(sig2)

* Get N and k
display "N = " e(N)
display "N_g = " e(N_g)
display "rank = " e(rank)
display "j0 = " e(j0)

* Get per-entity residuals (full system, as used in _ARTests)
* We need the diff-only residuals, which are stored in the full residual
predict pe, resid

* For entity=1, display diff residuals at each time
forvalues e=1/3 {
    display "Entity " `e' ":"
    forvalues t=0/4 {
        * The diff equation observation for entity e at time t
        * In the stacked data: diff rows are 1..T, level rows are T+1..2T
        * Not easily extractable from predict
    }
}

* Let's also manually compute AR1 and compare
* AR1 = Σ Σ e_it * e_i,t-1 / sqrt(...)
* We need the per-entity sums

* Actually let me try a different approach: use the stored estimates
* e(b) has the coefficients
matrix b = e(b)
matrix list b

* e(V) has the variance
matrix V = e(V)
matrix list V

* e(m2VZXA) and e(pV_ar) are not stored by xtabond2
* But we can reconstruct them

exit
