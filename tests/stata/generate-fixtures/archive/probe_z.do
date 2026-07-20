clear all
log using "C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\probe_z_stata.log", replace text
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear
xtset entity time
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                       gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) twostep small
di "j0 = " e(j0)
di "rank = " e(rank)
di "N = " e(N)
capture confirm matrix e(Z)
if _rc == 0 {
    matrix ZZ = e(Z)
    di "Z rows = " rowsof(ZZ) " Z cols = " colsof(ZZ)
    clear
    svmat double ZZ, names(col)
    export delimited using "C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\z_dump.csv", replace
    di "exported Z"
}
else {
    di "no e(Z)"
}
log close
