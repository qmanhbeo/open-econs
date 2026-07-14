* Simple test - output to file
clear all
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_logit.csv", clear

logit y x1 x2
margins, dydx(x1 x2)

* Export using postfile
tempname memhold
postfile `memhold' str20 var value using "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\margins_out.dta", replace
post `memhold' ("x1") (r(b)[1,1])
post `memhold' ("x2") (r(b)[1,2])
postfile `memhold' close

display "Done - x1=" r(b)[1,1] " x2=" r(b)[1,2]
exit