*! diag_estat.do — diagnostic parity fixtures (White / Breusch-Godfrey /
*!   Cook's D / leverage / dfbeta / Ljung-Box) for OLS y ~ x1 + x2.
clear all
set more off
log using "C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\diag_estat_run.log", text replace
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_ols.csv", clear
regress y x1 x2

* Breusch-Godfrey requires a tsset time variable (uses obs order).
gen long __t = _n
tsset __t

* ---- White heteroskedasticity (IMPORTANT: Stata 17 MP does NOT support
*      `estat hettest, white`; use `estat imtest, white`) ----
estat imtest, white
scalar s_white_chi2 = r(chi2)
scalar s_white_p    = r(p)
scalar s_white_df   = r(df)

* ---- Breusch-Godfrey (lags=1 and lags=2) ----
estat bgodfrey
* estat bgodfrey returns r(chi2)/r(p)/r(df) as 1x1 MATRICES, not scalars.
matrix m_bg = r(chi2)
matrix m_bgp = r(p)
scalar s_bg_chi2 = m_bg[1,1]
scalar s_bg_p    = m_bgp[1,1]
estat bgodfrey, lags(2)
matrix m_bg2 = r(chi2)
matrix m_bg2p = r(p)
scalar s_bg_chi2_2 = m_bg2[1,1]
scalar s_bg_p_2    = m_bg2p[1,1]

* ---- Cook's distance & leverage (save full vectors) ----
predict double cooksd, cooksd
predict double lev, leverage
preserve
keep cooksd
save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\diag_estat_cooks.dta", replace
restore
preserve
keep lev
save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\diag_estat_lev.dta", replace
restore

* ---- DFBETAS ----
* Stata's `dfbeta` command drops the constant by default and cannot be made to
* include it, so we replicate open_econs' standardized DFBETAS (b_j - b_{j(-i)})
* / SE_j(-i), leave-one-out) exactly in Mata for all 3 params, in the SAME
* parameter order open_econs uses: _dfbeta_1 = Intercept(const), _dfbeta_2 = x1,
* _dfbeta_3 = x2. This is the faithful Stata-side ground truth for r.dfbetas().
gen double _dfbeta_1 = .
gen double _dfbeta_2 = .
gen double _dfbeta_3 = .
mata:
st_view(y=., ., "y")
st_view(X=., ., ("x1","x2"))
st_view(D=., ., ("_dfbeta_1","_dfbeta_2","_dfbeta_3"))
n = rows(y); k = 2
Xf = (J(n,1,1), X)
XtX = quadcross(Xf, Xf)
XtX_inv = luinv(XtX)
b = XtX_inv * quadcross(Xf, y)
e = y - Xf * b
s2 = (e'*e) / (n - (k+1))
h = diagonal(Xf * XtX_inv * Xf')
for (i=1; i<=n; i++) {
    hi = h[i]
    d  = XtX_inv * Xf[i,.]' * (e[i] / (1 - hi))
    bminus = b - d
    s2_i = ((e'*e) - (e[i]^2) / (1 - hi)) / (n - (k+1) - 1)
    se = sqrt(s2_i) * sqrt(diagonal(XtX_inv))'
    D[i,.] = ((b - bminus)' :/ se)
}
end
preserve
keep _dfbeta_1 _dfbeta_2 _dfbeta_3
save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\diag_estat_dfbeta.dta", replace
restore

* ---- Ljung-Box (Stata has no `estat` LB on regress residuals; `wntestq`
*      is Stata's Ljung-Box Q / Portmanteau test) ----
predict double resid, resid
wntestq resid, lags(1)
* wntestq returns r(stat) (the Q statistic), r(p), r(df) -- NOT r(Q).
scalar s_lb_Q = r(stat)
scalar s_lb_p = r(p)

* ---- Scalar output dataset (name/value style) ----
clear
set obs 9
gen str20 name  = ""
gen double value = .

replace name = "white_chi2" in 1
replace name = "white_p"    in 2
replace name = "white_df"   in 3
replace name = "bg_chi2"    in 4
replace name = "bg_p"       in 5
replace name = "bg_chi2_2"  in 6
replace name = "bg_p_2"     in 7
replace name = "lb_Q"       in 8
replace name = "lb_p"       in 9

replace value = s_white_chi2 in 1
replace value = s_white_p    in 2
replace value = s_white_df   in 3
replace value = s_bg_chi2    in 4
replace value = s_bg_p       in 5
replace value = s_bg_chi2_2  in 6
replace value = s_bg_p_2     in 7
replace value = s_lb_Q       in 8
replace value = s_lb_p       in 9

save "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\diag_estat.dta", replace
