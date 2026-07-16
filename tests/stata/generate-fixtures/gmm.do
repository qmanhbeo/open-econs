*! gmm.do — Linear GMM parity fixtures for open-econs gmm() vs Stata gmm.
*! Stata/MP 17.0, Stata `gmm` command.
*! OE's gmm() includes intercept as its own instrument in Z.  To match,
*! Stata moment conditions must also include 1*(y-Xb) explicitly.
*! 8 specifications: exactly-ID × {1s,2s} × {nr,r} + overID × {1s,2s} × {nr,r}
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_gmm.csv", clear

postfile handle str32 name double value ///
    using "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\gmm.dta", replace

*==============================================================================
* EXACTLY-IDENTIFIED  (3 instruments = 3 parameters: intercept, z1, z2)
* OE formula: y ~ x1 + x2 | z1 + z2
* Z = [intercept, z1, z2], X = [intercept, x1, x2]
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
*==============================================================================

*--- Flavour 5: over-ID, one-step, non-robust ---
gmm (1*(y - {b0} - {b1}*x1 - {b2}*x2)) (z1*(y - {b0} - {b1}*x1 - {b2}*x2)) (z2*(y - {b0} - {b1}*x1 - {b2}*x2)) (z3*(y - {b0} - {b1}*x1 - {b2}*x2)) (z4*(y - {b0} - {b1}*x1 - {b2}*x2)) (z5*(y - {b0} - {b1}*x1 - {b2}*x2)), winitial(identity) onestep
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
gmm (1*(y - {b0} - {b1}*x1 - {b2}*x2)) (z1*(y - {b0} - {b1}*x1 - {b2}*x2)) (z2*(y - {b0} - {b1}*x1 - {b2}*x2)) (z3*(y - {b0} - {b1}*x1 - {b2}*x2)) (z4*(y - {b0} - {b1}*x1 - {b2}*x2)) (z5*(y - {b0} - {b1}*x1 - {b2}*x2)), winitial(identity) twostep
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
gmm (1*(y - {b0} - {b1}*x1 - {b2}*x2)) (z1*(y - {b0} - {b1}*x1 - {b2}*x2)) (z2*(y - {b0} - {b1}*x1 - {b2}*x2)) (z3*(y - {b0} - {b1}*x1 - {b2}*x2)) (z4*(y - {b0} - {b1}*x1 - {b2}*x2)) (z5*(y - {b0} - {b1}*x1 - {b2}*x2)), winitial(identity) onestep vce(robust)
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
gmm (1*(y - {b0} - {b1}*x1 - {b2}*x2)) (z1*(y - {b0} - {b1}*x1 - {b2}*x2)) (z2*(y - {b0} - {b1}*x1 - {b2}*x2)) (z3*(y - {b0} - {b1}*x1 - {b2}*x2)) (z4*(y - {b0} - {b1}*x1 - {b2}*x2)) (z5*(y - {b0} - {b1}*x1 - {b2}*x2)), winitial(identity) twostep vce(robust)
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
