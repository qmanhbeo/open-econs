*! event_study.do — Event-study regression parity fixture
*!
*! Note: Stata rejects negative values in factor variables (e.g. event_time = -1).
*! We use `post if treated == 1` instead, because for treated units
*!   event_time = post - 1  ⇒  D(event_time = 0) = post,
*! so the model is algebraically identical.
clear all
set more off
import delimited "C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_event_study.csv", clear

* ===== Model 1: no covariates  ==============================================
quietly regress y post if treated == 1, vce(hc2)

* Store results in locals (foreach needs locals, not scalars)
local m1_N        = e(N)
local m1_df_r     = e(df_r)
local m1_r2       = e(r2)

local m1_coef_Intercept  = _b[_cons]
local m1_se_Intercept    = _se[_cons]
local m1_t_Intercept     = _b[_cons] / _se[_cons]
local m1_p_Intercept     = 2 * ttail(e(df_r), abs(`m1_t_Intercept'))
local m1_ci95l_Intercept = _b[_cons] - invttail(e(df_r), 0.025) * _se[_cons]
local m1_ci95u_Intercept = _b[_cons] + invttail(e(df_r), 0.025) * _se[_cons]

local m1_coef_post = _b[post]
local m1_se_post   = _se[post]
local m1_t_post    = _b[post] / _se[post]
local m1_p_post    = 2 * ttail(e(df_r), abs(`m1_t_post'))
local m1_ci95l_post = _b[post] - invttail(e(df_r), 0.025) * _se[post]
local m1_ci95u_post = _b[post] + invttail(e(df_r), 0.025) * _se[post]

* ===== Model 2: with covariate x ============================================
quietly regress y post x if treated == 1, vce(hc2)

local m2_N        = e(N)
local m2_df_r     = e(df_r)
local m2_r2       = e(r2)

local m2_coef_Intercept  = _b[_cons]
local m2_se_Intercept    = _se[_cons]
local m2_t_Intercept     = _b[_cons] / _se[_cons]
local m2_p_Intercept     = 2 * ttail(e(df_r), abs(`m2_t_Intercept'))
local m2_ci95l_Intercept = _b[_cons] - invttail(e(df_r), 0.025) * _se[_cons]
local m2_ci95u_Intercept = _b[_cons] + invttail(e(df_r), 0.025) * _se[_cons]

local m2_coef_post = _b[post]
local m2_se_post   = _se[post]
local m2_t_post    = _b[post] / _se[post]
local m2_p_post    = 2 * ttail(e(df_r), abs(`m2_t_post'))
local m2_ci95l_post = _b[post] - invttail(e(df_r), 0.025) * _se[post]
local m2_ci95u_post = _b[post] + invttail(e(df_r), 0.025) * _se[post]

local m2_coef_x = _b[x]
local m2_se_x   = _se[x]
local m2_t_x    = _b[x] / _se[x]
local m2_p_x    = 2 * ttail(e(df_r), abs(`m2_t_x'))
local m2_ci95l_x = _b[x] - invttail(e(df_r), 0.025) * _se[x]
local m2_ci95u_x = _b[x] + invttail(e(df_r), 0.025) * _se[x]

* ===== Store all locals in flat name-value dataset ===========================
clear
set obs 36
gen str30 name  = ""
gen double value = .

local i 1
local names m1_N m1_df_r m1_r2
local names `names' m1_coef_Intercept m1_se_Intercept m1_t_Intercept m1_p_Intercept
local names `names' m1_ci95l_Intercept m1_ci95u_Intercept
local names `names' m1_coef_post m1_se_post m1_t_post m1_p_post m1_ci95l_post m1_ci95u_post

local names `names' m2_N m2_df_r m2_r2
local names `names' m2_coef_Intercept m2_se_Intercept m2_t_Intercept m2_p_Intercept
local names `names' m2_ci95l_Intercept m2_ci95u_Intercept
local names `names' m2_coef_post m2_se_post m2_t_post m2_p_post m2_ci95l_post m2_ci95u_post
local names `names' m2_coef_x m2_se_x m2_t_x m2_p_x m2_ci95l_x m2_ci95u_x

foreach n of local names {
    replace name = "`n'" in `i'
    replace value = ``n'' in `i'
    local i = `i' + 1
}

save "C:\Users\manhn\Desktop\open-econs\tests\stata\do\event_study.dta", replace
