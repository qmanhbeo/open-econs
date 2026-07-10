* Simple test
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_logit.csv", clear
logit y x1 x2
margins, dydx(x1 x2)
* Display results to see what we're getting
return list
ereturn list