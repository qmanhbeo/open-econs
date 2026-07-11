*! cem_autocuts_sturges.do — CEM with Sturges autocuts, export full results
clear all
set more off

import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_cem_autocuts.csv", clear

* Run CEM with autocuts(sturges) (default)
cap noi cem x1 x2 x3, treatment(t) autocuts(sturges)
if _rc != 0 {
    di "ERROR: cem with autocuts(sturges) returned code " _rc
    exit _rc
}

* Summarize
tab cem_strata
tab cem_matched

* Save full dataset with CEM variables
save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\cem_autocuts_sturges.dta", replace
