*! oaxaca_three_fold.do — Oaxaca-Blinder three-fold (SSC: oaxaca)
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_oaxaca.csv", clear
oaxaca y edu age, by(female)

matrix b = e(b)

scalar s_gap = b[1,3]

clear
set obs 1
gen str20 name  = ""
gen double value = .
replace name = "gap" in 1
replace value = s_gap in 1

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\oaxaca_three_fold.dta", replace
