clear
import delimited using "tests/r/fixtures/inputs/ur_input.csv", clear
gen t = _n
tsset t
* DF-GLS mu (trend = "c") at Stata's Ng-Perron SIC/MAIC-selected lag (=1) and at
* the seq-t / AIC-comparable lag (=0).  Stata's default lag-selection is
* Ng-Perron (SIC/MAIC/seq-t); OE (arch) uses AIC.  We capture both the SIC/MAIC
* lag statistic (lag 1) and the lag-0 statistic so the OE-vs-Stata DF-GLS
* lag-selection gap (FUTURE_WORK TS-2) can be asserted as an xfail.
* r(results) columns: k MAIC SIC RMSE DFGLS ; r(cvalues): k DFGLS 1% 5% 10%.
dfgls y, notrend maxlag(1)
matrix R1 = r(results)
scalar stat_lag1 = R1[1,5]
dfgls y, notrend maxlag(0)
scalar stat_lag0 = r(dft0)
scalar cv5_lag0  = r(cv_5)
* Ng-Perron SIC / MAIC optimal lag for this series.
dfgls y, notrend
scalar siclag  = r(siclag)
scalar maiclag = r(maiclag)
clear
set obs 5
gen str20 name = ""
gen double value = .
replace name = "stat_siclag" in 1
replace value = stat_lag1     in 1
replace name = "stat_lag0"   in 2
replace value = stat_lag0     in 2
replace name = "cv5_lag0"    in 3
replace value = cv5_lag0      in 3
replace name = "siclag"      in 4
replace value = siclag        in 4
replace name = "maiclag"     in 5
replace value = maiclag       in 5
save "tests/stata/fixtures/expected/ur_dfgls_c.dta", replace
