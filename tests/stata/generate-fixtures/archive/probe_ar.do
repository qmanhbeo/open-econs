*! probe_ar.do — Extract per-entity residuals and instruments for AR test
clear all
set more off

import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear
xtset entity time

* System GMM: two-step non-robust
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                       gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) twostep small

* Get AR test: e(ar1) = [AR stat, pval]
matrix a1 = e(ar1)
matrix a2 = e(ar2)
display "AR1 stat = " a1[1,1]
display "AR1 pval = " a1[1,2]
display "AR2 stat = " a2[1,1]
display "AR2 pval = " a2[1,2]

* Get regression residuals (full system)
predict pe, resid
* predict does full system (diff + level)

* For entity 1 (entity==1), show diff-only residuals
forvalues t=1/5 {
    * Diff equation residual at time t
    * In system GMM, the diff equation residual = pe[diff_obs]
}

* Get e(sig2) etc
display "e(N) = " e(N)
display "e(N_g) = " e(N_g)
display "e(j0) = " e(j0)
display "e(rank) = " e(rank)
display "e(sig2) = " e(sig2)

exit
