*! balance_weighted.do — Weighted covariate balance reference (Stata ground truth).
*!
*! Every scalar below was produced by a real Stata command:
*!   - summarize with [iw=w] for weighted means and variances
*!   - regress with [iw=w] for weighted OLS t-stats and p-values
*!   - scalar arithmetic from those Stata-produced numbers for SMD and VR
*! No values were hand-typed, read from pstest's display, or computed outside Stata.
*!
*! The SMD is stored as a raw ratio (not pstest's ×100 %-bias convention) to match
*! open-econs's output scaling — the test code does not multiply by 100.
*!
*! Weight scheme (PSM convention): treated always get weight 1 (as per pstest's
*! internal override), control weights drawn from Unif[0.5, 2.5] with seed 20240711.
*! open-econs's balance() applies weights uniformly (no treated-weight override)
*! but because treated weight is already 1 by construction, the two conventions
*! produce identical results.

clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_psm.csv", clear

* ---- Generate weights (treated = 1, control ~ Unif[0.5, 2.5]) ----
set seed 20240711
gen w = cond(t == 1, 1, runiform() * 2 + 0.5)

* ---- Export dataset with weights so Python reads the exact same values ----
export delimited using "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_balance_weighted.csv", replace

* ===== x1 =====
* Weighted treated mean (all weights=1 → same as unweighted)
sum x1 if t == 1
scalar m1m_x1 = r(mean)
* Weighted control mean [iw = w]
sum x1 if t == 0 [iw = w]
scalar m0m_x1 = r(mean)
scalar diff_x1 = m1m_x1 - m0m_x1

* Unweighted full-sample variances (SMD denominator, Rosenbaum-Rubin convention)
sum x1 if t == 1
scalar v1u_x1 = r(Var)
sum x1 if t == 0
scalar v0u_x1 = r(Var)
scalar pooledsd_x1 = sqrt((v1u_x1 + v0u_x1) / 2)
scalar smd_x1 = diff_x1 / pooledsd_x1

* Weighted (matched-sample) variances (VR numerator/denominator)
sum x1 if t == 1 [iw = w]
scalar v1m_x1 = r(Var)
sum x1 if t == 0 [iw = w]
scalar v0m_x1 = r(Var)
scalar vr_x1 = v1m_x1 / v0m_x1

* Weighted OLS t-test (regress covar treat [iw = w])
reg x1 t [iw = w]
scalar tstat_x1 = _b[t] / _se[t]
scalar pval_x1 = 2 * ttail(e(df_r), abs(tstat_x1))

* ===== x2 =====
sum x2 if t == 1
scalar m1m_x2 = r(mean)
sum x2 if t == 0 [iw = w]
scalar m0m_x2 = r(mean)
scalar diff_x2 = m1m_x2 - m0m_x2

sum x2 if t == 1
scalar v1u_x2 = r(Var)
sum x2 if t == 0
scalar v0u_x2 = r(Var)
scalar pooledsd_x2 = sqrt((v1u_x2 + v0u_x2) / 2)
scalar smd_x2 = diff_x2 / pooledsd_x2

sum x2 if t == 1 [iw = w]
scalar v1m_x2 = r(Var)
sum x2 if t == 0 [iw = w]
scalar v0m_x2 = r(Var)
scalar vr_x2 = v1m_x2 / v0m_x2

reg x2 t [iw = w]
scalar tstat_x2 = _b[t] / _se[t]
scalar pval_x2 = 2 * ttail(e(df_r), abs(tstat_x2))

* ===== Save reference scalars =====
clear
set obs 22
gen name = ""
gen double value = .

replace name = "diff_x1" in 1
replace name = "smd_x1"  in 2
replace name = "vr_x1"   in 3
replace name = "tstat_x1" in 4
replace name = "pval_x1" in 5
replace name = "v1u_x1"  in 6
replace name = "v0u_x1"  in 7
replace name = "v1m_x1"  in 8
replace name = "v0m_x1"  in 9
replace name = "m1m_x1"  in 10
replace name = "m0m_x1"  in 11

replace name = "diff_x2" in 12
replace name = "smd_x2"  in 13
replace name = "vr_x2"   in 14
replace name = "tstat_x2" in 15
replace name = "pval_x2" in 16
replace name = "v1u_x2"  in 17
replace name = "v0u_x2"  in 18
replace name = "v1m_x2"  in 19
replace name = "v0m_x2"  in 20
replace name = "m1m_x2"  in 21
replace name = "m0m_x2"  in 22

replace value = diff_x1  in 1
replace value = smd_x1   in 2
replace value = vr_x1    in 3
replace value = tstat_x1 in 4
replace value = pval_x1  in 5
replace value = v1u_x1   in 6
replace value = v0u_x1   in 7
replace value = v1m_x1   in 8
replace value = v0m_x1   in 9
replace value = m1m_x1   in 10
replace value = m0m_x1   in 11

replace value = diff_x2  in 12
replace value = smd_x2   in 13
replace value = vr_x2    in 14
replace value = tstat_x2 in 15
replace value = pval_x2  in 16
replace value = v1u_x2   in 17
replace value = v0u_x2   in 18
replace value = v1m_x2   in 19
replace value = v0m_x2   in 20
replace value = m1m_x2   in 21
replace value = m0m_x2   in 22

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\balance_weighted.dta", replace
