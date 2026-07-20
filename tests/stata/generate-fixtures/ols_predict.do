*! ols_predict.do â€” OLS predictions
clear all
set more off
set type double
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_ols.csv", clear
regress y x1 x2
predict yhat, xb
keep if _n <= 10
keep yhat
save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\ols_predict.dta", replace
