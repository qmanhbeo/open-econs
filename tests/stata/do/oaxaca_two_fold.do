*! oaxaca_two_fold.do — Oaxaca-Blinder two-fold via pooled reference (SSC: oaxaca)
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_oaxaca.csv", clear
oaxaca y edu age, by(female) pooled

matrix b = e(b)
matrix V = e(V)
local cols = colsof(b)

scalar s_gap   = b[1,3]
scalar s_exp   = b[1,4]
scalar s_unexp = b[1,5]

clear
set obs 3
gen str20 name  = ""
gen double value = .
replace name = "gap"        in 1
replace name = "explained"  in 2
replace name = "unexplained" in 3
replace value = s_gap  in 1
replace value = s_exp  in 2
replace value = s_unexp in 3

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\oaxaca_two_fold.dta", replace
