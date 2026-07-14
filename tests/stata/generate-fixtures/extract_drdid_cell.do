clear all
log using "C:\Users\manhn\AppData\Local\Temp\opencode\extract.log", text replace
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv", clear
drop if entity >= 20

* Build gvar exactly as staggered_did.do: 0 = never, 3 = treated at time 3
gen gvar = 0
replace gvar = 3 if entity >= 10 & entity < 20

* gsel for cohort 3: never-treated (gvar==0) OR gvar==3
gen byte gsel = (gvar==0 | gvar==3)
gen byte tr  = (gvar==3)

* Cell (g=3, pre=2, post=3): baseline=2, post=3
preserve
keep if gsel & inlist(time,2,3)
drdid y x z, ivar(entity) time(time) treatment(tr) drimp stub(__) replace pscoretrim(0.995)
* __att is the per-observation influence function (raw)
keep entity time __att
rename __att att_cell33
save "tests/stata/fixtures/expected/drdid_cell33.dta", replace
restore

* Cell (g=3, pre=2, post=4)
preserve
keep if gsel & inlist(time,2,4)
drdid y x z, ivar(entity) time(time) treatment(tr) drimp stub(__) replace pscoretrim(0.995)
keep entity time __att
rename __att att_cell34
save "tests/stata/fixtures/expected/drdid_cell34.dta", replace
restore

di "done"
