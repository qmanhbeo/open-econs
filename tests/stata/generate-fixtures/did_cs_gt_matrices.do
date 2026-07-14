*! did_cs_gt_matrices.do ??? Extract csdid per-obs RIF + cell V/ATT (svmat)
clear all
set more off
log using "tests/stata/fixtures/expected/did_cs_gt_matrices.log", replace text

import delimited "tests/stata/fixtures/inputs/df_panel.csv", clear
drop if entity >= 20
gen gvar = 0
replace gvar = 3 if entity >= 10 & entity < 20

* saverif() dumps the dataset WITH the per-obs RIF variables (_g3_*) to disk.
csdid y x z, ivar(entity) time(time) gvar(gvar) saverif(staggered_did_rif_save)

di as text "e(rif) = " e(rif)
di as text "e(b_attgt)[1,3] = " e(b_attgt)[1,3] " e(b_attgt)[1,4] = " e(b_attgt)[1,4]

matrix b_attgt = e(b_attgt)
svmat double b_attgt
export delimited using "tests/stata/fixtures/expected/did_cs_gt_battgt.csv", replace
drop b_attgt*

matrix V_attgt = e(V_attgt)
svmat double V_attgt
export delimited using "tests/stata/fixtures/expected/did_cs_gt_Vattgt.csv", replace
drop V_attgt*

matrix gtt = e(gtt)
svmat double gtt
export delimited using "tests/stata/fixtures/expected/did_cs_gt_gtt.csv", replace
drop gtt*

csdid_estat simple
di as text _newline "===== csdid_estat simple table ====="
matrix b_simple = e(b)
matrix list b_simple
matrix V_simple = e(V)
matrix list V_simple
svmat double b_simple
export delimited using "tests/stata/fixtures/expected/did_cs_gt_bsimple.csv", replace
drop b_simple*
svmat double V_simple
export delimited using "tests/stata/fixtures/expected/did_cs_gt_Vsimple.csv", replace
drop V_simple*

use staggered_did_rif_save.dta, clear
ds _g3_*
keep entity time gvar _g3_*
export delimited using "tests/stata/fixtures/expected/did_cs_gt_rif.csv", replace

di as text "DONE"
log close
