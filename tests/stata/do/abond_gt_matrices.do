*! abond_gt_matrices.do - Extract per-obs X, Y, Z, H matrices from Stata (svmat)
clear all
set more off
log using "C:\Users\manhn\Desktop\open-econs\tests\stata\do\abond_gt_matrices.log", replace text

import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv", clear
xtset entity time

*--- Target command: collapsed, one-step, non-robust, nolevel, small ---
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel small svmat

di as text "=== e(b) ==="
matrix list e(b)
di as text "=== e(V) ==="
matrix list e(V)
di as text "e(N)=" e(N) " e(j)=" e(j) " e(sig2)=" e(sig2)

* Pull the saved matrices into memory and export to CSV for Python comparison
matrix Hmat = e(H)
svmat double Hmat, names(col)
export delimited using "C:\Users\manhn\Desktop\open-econs\tests\stata\do\abond_gt_H.csv", replace
drop Hmat*

matrix Xmat = e(X)
svmat double Xmat, names(col)
export delimited using "C:\Users\manhn\Desktop\open-econs\tests\stata\do\abond_gt_X.csv", replace
drop Xmat*

matrix Ymat = e(Y)
svmat double Ymat, names(col)
export delimited using "C:\Users\manhn\Desktop\open-econs\tests\stata\do\abond_gt_Y.csv", replace
drop Ymat*

capture confirm matrix e(Z)
if _rc == 0 {
    matrix Zmat = e(Z)
    svmat double Zmat, names(col)
    export delimited using "C:\Users\manhn\Desktop\open-econs\tests\stata\do\abond_gt_Z.csv", replace
    drop Zmat*
    di as text "e(Z) SAVED"
}
else {
    di as text "e(Z) NOT AVAILABLE (space-favoring mode)"
}

matrix IMmat = e(ideqt)
svmat double IMmat, names(col)
export delimited using "C:\Users\manhn\Desktop\open-econs\tests\stata\do\abond_gt_ideqt.csv", replace
drop IMmat*

di as text "DONE"
