*! gmm.do — Linear GMM parity fixtures for open-econs gmm() vs Stata gmm.
*! Stata/MP 17.0, Stata `gmm` command.
*!
*! Exactly-identified cases: expression form with winitial(identity) (already
*! matches OE to machine epsilon — identity weighting gives the same
*! estimator as (Z'Z)^{-1} when L == p).
*!
*! Over-identified cases: single-equation form with instruments() and
*! winitial(unadjusted).  This gives the standard 2SLS estimator
*! (b = (X'Z(Z'Z)^{-1}Z'X)^{-1} X'Z(Z'Z)^{-1}Z'Y), matching OE's
*! closed-form two-step GMM coefficients exactly.  The previous fixture
*! used expression-form moment conditions with winitial(identity), which
*! minimizes (Y-Xb)'ZZ'(Y-Xb) — a different objective that does NOT
*! give 2SLS in the overidentified case (confirmed 2026-07-17).
*!
*! NOTE on two-step SEs: Stata's `gmm` does NOT apply the Windmeijer
*! (2005) finite-sample correction for two-step robust VCE.  OE's gmm()
*! always applies Windmeijer.  Two-step robust SEs therefore diverge
*! (~15%) and are NOT valid parity targets for SE assertions.
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_gmm.csv", clear

postfile handle str32 name double value ///
    using "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\gmm.dta", replace

*==============================================================================
* EXACTLY-IDENTIFIED  (3 instruments = 3 parameters: intercept, z1, z2)
* OE formula: y ~ x1 + x2 | z1 + z2
* Z = [intercept, z1, z2], X = [intercept, x1, x2]
* Expression form: each equation has only _cons, winitial(identity) gives W=I.
* For L==p the identity-weighted and (Z'Z)^{-1}-weighted estimators coincide.
*==============================================================================

*--- Flavour 1: exactly-ID, one-step, non-robust ---
gmm (1*(y - {b0} - {b1}*x1 - {b2}*x2)) (z1*(y - {b0} - {b1}*x1 - {b2}*x2)) (z2*(y - {b0} - {b1}*x1 - {b2}*x2)), winitial(identity) onestep
matrix b = e(b)
matrix V = e(V)
post handle ("b0_eid_1s_nr")    (b[1,1])
post handle ("b1_eid_1s_nr")    (b[1,2])
post handle ("b2_eid_1s_nr")    (b[1,3])
post handle ("se0_eid_1s_nr")   (sqrt(V[1,1]))
post handle ("se1_eid_1s_nr")   (sqrt(V[2,2]))
post handle ("se2_eid_1s_nr")   (sqrt(V[3,3]))
post handle ("N_eid_1s_nr")     (e(N))
post handle ("J_eid_1s_nr")     (e(J))
post handle ("Jdf_eid_1s_nr")   (e(J_df))

*--- Flavour 2: exactly-ID, two-step, non-robust ---
gmm (1*(y - {b0} - {b1}*x1 - {b2}*x2)) (z1*(y - {b0} - {b1}*x1 - {b2}*x2)) (z2*(y - {b0} - {b1}*x1 - {b2}*x2)), winitial(identity) twostep
matrix b = e(b)
matrix V = e(V)
post handle ("b0_eid_2s_nr")    (b[1,1])
post handle ("b1_eid_2s_nr")    (b[1,2])
post handle ("b2_eid_2s_nr")    (b[1,3])
post handle ("se0_eid_2s_nr")   (sqrt(V[1,1]))
post handle ("se1_eid_2s_nr")   (sqrt(V[2,2]))
post handle ("se2_eid_2s_nr")   (sqrt(V[3,3]))
post handle ("N_eid_2s_nr")     (e(N))
post handle ("J_eid_2s_nr")     (e(J))
post handle ("Jdf_eid_2s_nr")   (e(J_df))

*--- Flavour 3: exactly-ID, one-step, robust ---
gmm (1*(y - {b0} - {b1}*x1 - {b2}*x2)) (z1*(y - {b0} - {b1}*x1 - {b2}*x2)) (z2*(y - {b0} - {b1}*x1 - {b2}*x2)), winitial(identity) onestep vce(robust)
matrix b = e(b)
matrix V = e(V)
post handle ("b0_eid_1s_r")     (b[1,1])
post handle ("b1_eid_1s_r")     (b[1,2])
post handle ("b2_eid_1s_r")     (b[1,3])
post handle ("se0_eid_1s_r")    (sqrt(V[1,1]))
post handle ("se1_eid_1s_r")    (sqrt(V[2,2]))
post handle ("se2_eid_1s_r")    (sqrt(V[3,3]))
post handle ("N_eid_1s_r")      (e(N))
post handle ("J_eid_1s_r")      (e(J))
post handle ("Jdf_eid_1s_r")    (e(J_df))

*--- Flavour 4: exactly-ID, two-step, robust ---
gmm (1*(y - {b0} - {b1}*x1 - {b2}*x2)) (z1*(y - {b0} - {b1}*x1 - {b2}*x2)) (z2*(y - {b0} - {b1}*x1 - {b2}*x2)), winitial(identity) twostep vce(robust)
matrix b = e(b)
matrix V = e(V)
post handle ("b0_eid_2s_r")     (b[1,1])
post handle ("b1_eid_2s_r")     (b[1,2])
post handle ("b2_eid_2s_r")     (b[1,3])
post handle ("se0_eid_2s_r")    (sqrt(V[1,1]))
post handle ("se1_eid_2s_r")    (sqrt(V[2,2]))
post handle ("se2_eid_2s_r")    (sqrt(V[3,3]))
post handle ("N_eid_2s_r")      (e(N))
post handle ("J_eid_2s_r")      (e(J))
post handle ("Jdf_eid_2s_r")    (e(J_df))

*==============================================================================
* OVERIDENTIFIED  (6 instruments = z0(=intercept), z1..z5, 3 parameters)
* OE formula: y ~ x1 + x2 | z1 + z2 + z3 + z4 + z5
* Z = [intercept, z1, z2, z3, z4, z5], X = [intercept, x1, x2]
*
* Single-equation form with instruments(): Stata adds _cons automatically,
* giving the same Z as OE.  winitial(unadjusted) uses (Z'Z)^{-1}, giving
* the standard 2SLS one-step estimator.  Two-step uses efficient S^{-1}.
*==============================================================================

*--- Flavour 5: over-ID, one-step, non-robust ---
gmm (y - {b0} - {b1}*x1 - {b2}*x2), instruments(z1 z2 z3 z4 z5) winitial(unadjusted) onestep
matrix b = e(b)
matrix V = e(V)
post handle ("b0_oid_1s_nr")    (b[1,1])
post handle ("b1_oid_1s_nr")    (b[1,2])
post handle ("b2_oid_1s_nr")    (b[1,3])
post handle ("se0_oid_1s_nr")   (sqrt(V[1,1]))
post handle ("se1_oid_1s_nr")   (sqrt(V[2,2]))
post handle ("se2_oid_1s_nr")   (sqrt(V[3,3]))
post handle ("N_oid_1s_nr")     (e(N))
post handle ("J_oid_1s_nr")     (e(J))
post handle ("Jdf_oid_1s_nr")   (e(J_df))

*--- Flavour 6: over-ID, two-step, non-robust ---
gmm (y - {b0} - {b1}*x1 - {b2}*x2), instruments(z1 z2 z3 z4 z5) winitial(unadjusted) twostep
matrix b = e(b)
matrix V = e(V)
post handle ("b0_oid_2s_nr")    (b[1,1])
post handle ("b1_oid_2s_nr")    (b[1,2])
post handle ("b2_oid_2s_nr")    (b[1,3])
post handle ("se0_oid_2s_nr")   (sqrt(V[1,1]))
post handle ("se1_oid_2s_nr")   (sqrt(V[2,2]))
post handle ("se2_oid_2s_nr")   (sqrt(V[3,3]))
post handle ("N_oid_2s_nr")     (e(N))
post handle ("J_oid_2s_nr")     (e(J))
post handle ("Jdf_oid_2s_nr")   (e(J_df))

*--- Flavour 7: over-ID, one-step, robust ---
gmm (y - {b0} - {b1}*x1 - {b2}*x2), instruments(z1 z2 z3 z4 z5) winitial(unadjusted) onestep vce(robust)
matrix b = e(b)
matrix V = e(V)
post handle ("b0_oid_1s_r")     (b[1,1])
post handle ("b1_oid_1s_r")     (b[1,2])
post handle ("b2_oid_1s_r")     (b[1,3])
post handle ("se0_oid_1s_r")    (sqrt(V[1,1]))
post handle ("se1_oid_1s_r")    (sqrt(V[2,2]))
post handle ("se2_oid_1s_r")    (sqrt(V[3,3]))
post handle ("N_oid_1s_r")      (e(N))
post handle ("J_oid_1s_r")      (e(J))
post handle ("Jdf_oid_1s_r")    (e(J_df))

*--- Flavour 8: over-ID, two-step, robust ---
gmm (y - {b0} - {b1}*x1 - {b2}*x2), instruments(z1 z2 z3 z4 z5) winitial(unadjusted) twostep vce(robust)
matrix b = e(b)
matrix V = e(V)
post handle ("b0_oid_2s_r")     (b[1,1])
post handle ("b1_oid_2s_r")     (b[1,2])
post handle ("b2_oid_2s_r")     (b[1,3])
post handle ("se0_oid_2s_r")    (sqrt(V[1,1]))
post handle ("se1_oid_2s_r")    (sqrt(V[2,2]))
post handle ("se2_oid_2s_r")    (sqrt(V[3,3]))
post handle ("N_oid_2s_r")      (e(N))
post handle ("J_oid_2s_r")      (e(J))
post handle ("Jdf_oid_2s_r")    (e(J_df))

*--- Metadata row ---
post handle ("stata_version") (17.0)

postclose handle
