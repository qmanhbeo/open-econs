*! abond_ground_truth.do - Extract exact Stata internals for collapsed one-step
clear all
set more off
log using "C:\Users\manhn\Desktop\open-econs\tests\stata\do\abond_ground_truth.log", replace text

import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv", clear
xtset entity time

*--- COLLAPSED Run B: lag(2 4) 1-step, nolevel, small ---
di as text "=== COLLAPSED Run B: lag(2 4) 1-step (collapse in gmm) ==="
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel small

di as text ""
di as text "=== e(b) ==="
matrix list e(b)

di as text ""
di as text "=== e(V) ==="
matrix list e(V)

di as text ""
di as text "=== Stored scalars ==="
di as text "e(N) = " e(N)
di as text "e(j) = " e(j)
di as text "e(sig2) = " e(sig2)

di as text ""
di as text "=== adopath ==="
adopath

di as text ""
di as text "=== Which glsaccum ==="
capture which glsaccum

di as text ""
di as text "=== xtabond2.ado location ==="
which xtabond2

log close
