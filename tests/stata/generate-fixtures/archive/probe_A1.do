clear all
log using "C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\probe_A1.log", replace text
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear
xtset entity time
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                       gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) twostep small
matrix A1 = e(A1)
matrix A2 = e(A2)
matrix b = e(b)
clear
svmat double A1, names(col)
export delimited using "C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\A1.csv", replace
clear
svmat double A2, names(col)
export delimited using "C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\A2.csv", replace
di "saved A1,A2"
log close
