*! cem_autocuts_ss.do — CEM with SS autocuts, export full results
clear all
set more off

import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_cem_autocuts.csv", clear

cap noi cem x1 x2 x3, treatment(t) autocuts(ss)
if _rc != 0 {
    di "ERROR: cem with autocuts(ss) returned code " _rc
    exit _rc
}

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\cem_autocuts_ss.dta", replace
