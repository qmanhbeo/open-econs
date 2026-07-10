* Simple test - just output margins
clear all
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_logit.csv", clear

logit y x1 x2
margins, dydx(x1 x2)

* Try to get the margin values - use r(b) which holds the point estimates
matrix list r(b)
display "x1 margin = " r(b)[1,1]
display "x2 margin = " r(b)[1,2]