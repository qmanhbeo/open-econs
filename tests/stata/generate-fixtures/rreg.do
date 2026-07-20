*! rreg.do â€” Stata rreg (robust regression, Tukey bisquare M-estimator) parity fixture.
*! Stata/MP 17.0, `rreg` command.
*!
*! Stata `rreg y x1 x2` = robust regression: M-estimator with Tukey bisquare
*! (biweight) psi, c=4.685, Huber initial (k=1.345), IRLS. Reports e(b) and a
*! robust (sandwich) e(V) by default. We dump e(b), e(se), and e(V) to a .dta.
clear all
set more off
set type double
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\rreg_input.csv", clear

rreg y x1 x2

matrix b = e(b)
matrix V = e(V)
matrix list b
matrix list V

postfile handle str32 name double value ///
    using "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\rreg.dta", replace

* NOTE: Stata rreg e(b) is ordered [x1, x2, _cons].
post handle ("b_x1")     (b[1,1])
post handle ("b_x2")     (b[1,2])
post handle ("b_cons")   (b[1,3])
post handle ("se_x1")    (sqrt(V[1,1]))
post handle ("se_x2")    (sqrt(V[2,2]))
post handle ("se_cons")  (sqrt(V[3,3]))
post handle ("V11")      (V[1,1])
post handle ("V12")      (V[1,2])
post handle ("V13")      (V[1,3])
post handle ("V22")      (V[2,2])
post handle ("V23")      (V[2,3])
post handle ("V33")      (V[3,3])
post handle ("V21")      (V[2,1])
post handle ("V31")      (V[3,1])
post handle ("V32")      (V[3,2])
post handle ("N")        (e(N))
post handle ("rss")      (e(rss))
post handle ("rmse")     (e(rmse))

postclose handle
