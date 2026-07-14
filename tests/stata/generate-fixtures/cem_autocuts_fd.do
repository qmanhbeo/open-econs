*! cem_autocuts_fd.do — CEM with FD autocuts, export full results
clear all
set more off

import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_cem_autocuts.csv", clear

cap noi cem x1 x2 x3, treatment(t) autocuts(fd)
if _rc != 0 {
    di "ERROR: cem with autocuts(fd) returned code " _rc
    exit _rc
}

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\cem_autocuts_fd.dta", replace
