clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_gmm.csv", clear

gmm (y - {b0} - {b1}*x1 - {b2}*x2), instruments(z1 z2 z3 z4 z5) ///
    winitial(unadjusted) twostep vce(robust)

matrix b = e(b)
matrix V = e(V)

postfile handle str32 name double value ///
    using "C:\Users\manhn\Desktop\open-econs\tests/stata/fixtures/expected\gmm_diag2.dta", replace

post handle ("b0") (b[1,1])
post handle ("b1") (b[1,2])
post handle ("b2") (b[1,3])
post handle ("se0") (sqrt(V[1,1]))
post handle ("se1") (sqrt(V[2,2]))
post handle ("se2") (sqrt(V[3,3]))

* Reconstruct candidate VCEs from stored matrices in Stata itself
matrix Gmat = e(G)
matrix Wf = e(W_final)
matrix Smat = e(S)
matrix Wmat = e(W)

* (1) inv(G' Wf G)
matrix A1 = Gmat' * Wf * Gmat
matrix V1 = inv(A1)
post handle ("se0_recon_GWfG") (sqrt(V1[1,1]))
post handle ("se1_recon_GWfG") (sqrt(V1[2,2]))
post handle ("se2_recon_GWfG") (sqrt(V1[3,3]))

* (2) full sandwich: inv(G'WfG) G'Wf Wf G inv(G'WfG)
matrix mid = Gmat' * Wf * Wf * Gmat
matrix V2 = V1 * mid * V1
post handle ("se0_recon_full") (sqrt(V2[1,1]))
post handle ("se1_recon_full") (sqrt(V2[2,2]))
post handle ("se2_recon_full") (sqrt(V2[3,3]))

* (3) inv(G' Sinv G)
matrix Sinv = inv(Smat)
matrix A3 = Gmat' * Sinv * Gmat
matrix V3 = inv(A3)
post handle ("se0_recon_GSinvg") (sqrt(V3[1,1]))
post handle ("se1_recon_GSinvg") (sqrt(V3[2,2]))
post handle ("se2_recon_GSinvg") (sqrt(V3[3,3]))

* (4) inv(G' W G) with W = e(W)
matrix A4 = Gmat' * Wmat * Gmat
matrix V4 = inv(A4)
post handle ("se0_recon_GWG") (sqrt(V4[1,1]))
post handle ("se1_recon_GWG") (sqrt(V4[2,2]))
post handle ("se2_recon_GWG") (sqrt(V4[3,3]))

postclose handle
