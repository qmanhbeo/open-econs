* Debug margins
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_logit.csv", clear

* Run logit
logit y x1 x2

* Show coefficients
display "Coefficients:"
display "x1 = " _b[x1]
display "x2 = " _b[x2]

* Show margins with all options
display "Margins (default - dydx):"
margins, dydx(x1 x2)

display "Margins at means:"
margins, dydx(x1 x2) atmeans

display "Margins at mean of x1, median of x2:"
margins, dydx(x1 x2) at(x1=mean x2=median)

* Export marginal effects
matrix m = e(b)'
scalar s_me1 = m[1,1]
scalar s_me2 = m[2,1]

display "Matrix values: me_x1 = " s_me1 " me_x2 = " s_me2