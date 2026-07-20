clear all
log using "C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\probe_eret.log", replace text
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear
xtset entity time
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                       gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) twostep small
di "================ ereturn list ================"
ereturn list
di "================ matrices ================"
foreach m in Z Ze W A Gmm iv b V S {
    capture confirm matrix e(`m')
    if _rc == 0 {
        di "FOUND e(`m')  rows=" rowsof(e(`m')) " cols=" colsof(e(`m'))
    }
}
log close
