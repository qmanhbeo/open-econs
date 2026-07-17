*! gmm_diag.do — Extract internal matrices from two-step robust GMM
*! for element-by-element comparison with OE's computation.

clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_gmm.csv", clear

*==============================================================================
* Case: over-identified, two-step, robust, winitial(unadjusted)
*==============================================================================
gmm (y - {b0} - {b1}*x1 - {b2}*x2), instruments(z1 z2 z3 z4 z5) ///
    winitial(unadjusted) twostep vce(robust)

* Store b and V
matrix b = e(b)
matrix V = e(V)

* Store the weighting matrix W (final weighting matrix used in step 2)
* Stata stores this in e(W)
capture matrix Wfinal = e(W)

* Store the S matrix (moment condition covariance)
* Stata may store this in e(S)
capture matrix Smat = e(S)

* Count dimensions
local kp = colsof(b)
local kl = 0
capture local kl = colsof(Wfinal)
local kn = e(N)

* Write b
postfile handle str32 name double value ///
    using "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\gmm_diag.dta", replace

post handle ("b0")        (b[1,1])
post handle ("b1")        (b[1,2])
post handle ("b2")        (b[1,3])
post handle ("se0")       (sqrt(V[1,1]))
post handle ("se1")       (sqrt(V[2,2]))
post handle ("se2")       (sqrt(V[3,3]))
post handle ("N")          (`kn')
post handle ("kp")         (`kp')
post handle ("kl")         (`kl')

* Post diagonal of V
forvalues i = 1/`kp' {
    post handle ("Vdiag_`i'")  (V[`i',`i'])
}

* Post full V as flat elements
forvalues i = 1/`kp' {
    forvalues j = 1/`kp' {
        post handle ("V_`i'_`j'")  (V[`i',`j'])
    }
}

* Post W final (if available)
if `kl' > 0 {
    forvalues i = 1/`kl' {
        forvalues j = 1/`kl' {
            post handle ("W_`i'_`j'")  (Wfinal[`i',`j'])
        }
    }
}

* Post S (if available)
capture {
    local sl = rowsof(Smat)
    if `sl' > 0 {
        forvalues i = 1/`sl' {
            forvalues j = 1/`sl' {
                post handle ("S_`i'_`j'")  (Smat[`i',`j'])
            }
        }
    }
}

* Now also get the one-step VCE (V1) and residuals-derived S
* by running one-step to compare
gmm (y - {b0} - {b1}*x1 - {b2}*x2), instruments(z1 z2 z3 z4 z5) ///
    winitial(unadjusted) onestep vce(robust)

matrix b1step = e(b)
matrix V1step = e(V)

post handle ("b1s_0")      (b1step[1,1])
post handle ("b1s_1")      (b1step[1,2])
post handle ("b1s_2")      (b1step[1,3])
post handle ("se1s_0")     (sqrt(V1step[1,1]))
post handle ("se1s_1")     (sqrt(V1step[2,2]))
post handle ("se1s_2")     (sqrt(V1step[3,3]))

* Post the one-step W (initial weighting matrix)
capture matrix W1 = e(W)
capture {
    local wl = rowsof(W1)
    if `wl' > 0 {
        forvalues i = 1/`wl' {
            forvalues j = 1/`wl' {
                post handle ("W1_`i'_`j'")  (W1[`i',`j'])
            }
        }
    }
}

postclose handle
