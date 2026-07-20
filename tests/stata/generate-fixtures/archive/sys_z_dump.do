clear all
set more off
log using "C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\sys_z_dump.log", replace text

import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear
xtset entity time

xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                       gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) twostep small svmat

di as text "=== e(b) ==="
matrix list e(b)
di as text "e(N)=" e(N) " e(j0)=" e(j0) " e(sig2)=" e(sig2)

capture confirm matrix e(Z)
if _rc == 0 {
    di as text "e(Z) AVAILABLE: rows=" rowsof(e(Z)) " cols=" colsof(e(Z))
    matrix Zmat = e(Z)
    svmat double Zmat
    export delimited using "C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\sys_Z.csv", replace
    di as text "exported sys_Z.csv"
}
else {
    di as text "no e(Z) via svmat"
}

capture confirm matrix e(X)
if _rc == 0 {
    matrix Xmat = e(X)
    svmat double Xmat
    export delimited using "C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\sys_X.csv", replace
    di as text "exported sys_X.csv"
}
capture confirm matrix e(Y)
if _rc == 0 {
    matrix Ymat = e(Y)
    svmat double Ymat
    export delimited using "C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\sys_Y.csv", replace
    di as text "exported sys_Y.csv"
}
capture confirm matrix e(H)
if _rc == 0 {
    matrix Hmat = e(H)
    svmat double Hmat
    export delimited using "C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\sys_H.csv", replace
    di as text "exported sys_H.csv"
}
di as text "DONE"
log close
