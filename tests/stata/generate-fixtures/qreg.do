*! qreg.do — Quantile regression parity fixtures for open-econs quantile_reg().
*! Stata/MP 17.0.  Estimators: qreg (median + 0.25), sqreg, bsqreg.
*!
*! DESIGN
*!   qreg y x1 x2, quantile(tau)
*!     - Coefficients: Barrodale–Roberts simplex (== R rq(method="br")).
*!     - Default SE (e(V)): sparsity/Powell-kernel Hall–Sheather sandwich.
*!   sqreg  = simultaneous quantiles, bootstrap VCE (reps=20, seed fixed).
*!   bsqreg = single-quantile bootstrap VCE (reps=20, seed fixed).
*!
*! Two fixtures:
*!   qreg.dta       — qreg coefs + default SEs at tau in {0.25, 0.5}.
*!   qreg_boot.dta  — sqreg / bsqreg coefs + bootstrap SEs (set seed 20260719).
*!
*! Bootstrap NOTE: Stata's bootstrap RNG is not portable across platforms; the
*! committed .dta records Stata's own seeded values.  The Python side reproduces
*! coefficients (deterministic BR simplex) to 1e-6 and asserts bootstrap SEs to
*! a documented tolerance only (see test_stata_qreg.py).
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_qreg.csv", clear

*==============================================================================
* Fixture 1: qreg default SE at tau=0.5 (median, default) and tau=0.25
*==============================================================================
postfile handle str32 name double value ///
    using "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\qreg.dta", replace

*--- tau = 0.5 (median regression, Stata default quantile) ---
qreg y x1 x2, quantile(0.5)
matrix b = e(b)
matrix V = e(V)
post handle ("b0_q50")  (b[1,3])
post handle ("b1_q50")  (b[1,1])
post handle ("b2_q50")  (b[1,2])
post handle ("se0_q50") (sqrt(V[3,3]))
post handle ("se1_q50") (sqrt(V[1,1]))
post handle ("se2_q50") (sqrt(V[2,2]))
post handle ("N_q50")   (e(N))

*--- tau = 0.25 ---
qreg y x1 x2, quantile(0.25)
matrix b = e(b)
matrix V = e(V)
post handle ("b0_q25")  (b[1,3])
post handle ("b1_q25")  (b[1,1])
post handle ("b2_q25")  (b[1,2])
post handle ("se0_q25") (sqrt(V[3,3]))
post handle ("se1_q25") (sqrt(V[1,1]))
post handle ("se2_q25") (sqrt(V[2,2]))
post handle ("N_q25")   (e(N))

post handle ("stata_version") (17.0)
postclose handle

*==============================================================================
* Fixture 2: bootstrap variants (sqreg, bsqreg), fixed seed, reps=20
*==============================================================================
postfile bhandle str32 name double value ///
    using "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\qreg_boot.dta", replace

*--- bsqreg: single quantile (0.5) bootstrap, reps=20 ---
set seed 20260719
bsqreg y x1 x2, quantile(0.5) reps(20)
matrix b = e(b)
matrix V = e(V)
post bhandle ("bsq_b0_q50")  (b[1,3])
post bhandle ("bsq_b1_q50")  (b[1,1])
post bhandle ("bsq_b2_q50")  (b[1,2])
post bhandle ("bsq_se0_q50") (sqrt(V[3,3]))
post bhandle ("bsq_se1_q50") (sqrt(V[1,1]))
post bhandle ("bsq_se2_q50") (sqrt(V[2,2]))

*--- sqreg: simultaneous quantiles (0.25, 0.5, 0.75) bootstrap, reps=20 ---
set seed 20260719
sqreg y x1 x2, quantiles(0.25 0.5 0.75) reps(20)
matrix b = e(b)
matrix V = e(V)
* sqreg stores coefs blocked by quantile: q25 (x1 x2 _cons), q50 (...), q75 (...)
post bhandle ("sq_b0_q50")  (b[1,6])
post bhandle ("sq_b1_q50")  (b[1,4])
post bhandle ("sq_b2_q50")  (b[1,5])
post bhandle ("sq_se0_q50") (sqrt(V[6,6]))
post bhandle ("sq_se1_q50") (sqrt(V[4,4]))
post bhandle ("sq_se2_q50") (sqrt(V[5,5]))

postclose bhandle
