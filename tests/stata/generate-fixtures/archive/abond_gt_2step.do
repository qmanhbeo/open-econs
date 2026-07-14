*! abond_gt_2step.do - Extract b, V(SE), sig2, N, j, AR(1/2) from Stata (xtabond2 3.7.2)
clear all
set more off
log using "C:\Users\manhn\Desktop\open-econs\tests\stata\do\abond_gt_2step.log", replace text

* Force "favor speed" so svmat saves e(Z) as well as e(X)/e(Y)/e(H)
mata: mata set matafavor speed, perm

import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear
xtset entity time

foreach spec in onestep twostep robust twosteprobust {
    if "`spec'" == "onestep"         local cmd y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel small svmat
    if "`spec'" == "twostep"         local cmd y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel small twostep svmat
    if "`spec'" == "robust"          local cmd y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel small robust svmat
    if "`spec'" == "twosteprobust"   local cmd y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel small twostep robust svmat

    di as text "===== SPEC `spec' ====="
    xtabond2 `cmd'

    matrix b = e(b)
    matrix V = e(V)
    di as text "b  = " %18.10f b[1,1] " " %18.10f b[1,2] " " %18.10f b[1,3]
    di as text "se = " %18.10f sqrt(el(V,1,1)) " " %18.10f sqrt(el(V,2,2)) " " %18.10f sqrt(el(V,3,3))
    di as text "Vdiag = " %18.10f el(V,1,1) " " %18.10f el(V,2,2) " " %18.10f el(V,3,3)
    di as text "sig2 = " %18.10f e(sig2) "  N = " e(N) "  j = " e(j)
    di as text "ar1 = " %18.10f e(ar1) "  ar1p = " %18.10f e(ar1p)
    di as text "ar2 = " %18.10f e(ar2) "  ar2p = " %18.10f e(ar2p)
}
log close
