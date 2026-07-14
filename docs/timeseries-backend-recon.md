# Time-Series Backend Recon (`arch` / `statsmodels.tsa` vs Stata / R)

**Status:** Source-verified recon, pre-build. Written during the Step-3 source
recon for v1.1.0. **No wrapper code has been written yet.** This document records
the convention gaps discovered between the wrapped backends (`arch.unitroot`,
`statsmodels.tsa.arima.model.ARIMA`, `arch.arch_model`) and the Stata/R reference
tools, so the parity strategy can be decided **before** any code is committed.

**Standing-rule reminder:** wrapping is an implementation strategy, not a parity
exemption. Every gap below must be *either* root-caused-and-fixed,
corrected-for in the OE wrapper, or documented as a source-confirmed intentional
convention. None may be silently absorbed.

**Tools verified (all present):**
- Stata 17 base: `dfuller.ado` (v1.5.0), `pperron.ado` (v1.2.0), `dfgls.ado`
  (v1.2.3), `arima.ado`, `arch.ado`. **KPSS is NOT in Stata 17 base** (community
  SSC only). **No standalone `garch.ado`** — GARCH is `arch` with `garch()`.
- R 4.6.1: `urca` (ur.df / ur.pp / ur.kpss / ur.za) installed. `forecast` /
  `rugarch` **NOT installed** (defaults below are documented, not source-verified
  in this environment).
- `arch` 8.0.0: `unitroot` (ADF / DFGLS / PhillipsPerron / KPSS / ZivotAndrews /
  VarianceRatio), `arch_model`.
- `statsmodels` 0.14.6: `tsa.arima.model.ARIMA`.

---

## 1. Unit-root tests — critical-value-table vintage is the crux

### 1.1 ADF (`adf()`)

| Dimension | `arch.unitroot.ADF` | Stata `dfuller` | R `ur.df` |
|---|---|---|---|
| Regression | Δy = ρ·yₜ₋₁ + ΣβᵢΔyₜ₋ᵢ + (const/trend) | identical (case2=const, case4=trend, case1=none, case3=drift) | identical (`type` none/drift/trend) |
| Statistic | t on yₜ₋₁ (τ) | t on yₜ₋₁ ("Z(t)") — **only Z(t), never Z(ρ)** | τ (t on yₜ₋₁) |
| **Default lags** | **auto AIC**, `max_lags = 12·(n/100)^¼` (Schwert) | **0** (no automatic lag selection) | **fixed 1** (`selectlags="Fixed"`) |
| **Critical VALUES** | **MacKinnon (2010)** `tau_2010` response surface | **Fuller (1976)** finite-sample table (no citation string) | **Fuller-style** banded finite-sample table |
| **p-value** | **MacKinnon (1994)** response surface (`tau_small/large_p`) | **MacKinnon (1994)** (`MacP`, matching coeffs) | **none returned** |

**Conflict (real, must decide):** arch's displayed *critical values* come from
MacKinnon (2010); Stata prints Fuller (1976) tables; R prints banded Fuller.
These agree at large N (same asymptotic limit) but **diverge in small samples**.
arch's *p-value* matches Stata's "MacKinnon approximate p-value" (both MacKinnon
1994). So the **p-value is a clean cross-check anchor**, but the **printed CV
table is not identical** to Stata's. This is the exact silent-mismatch risk the
wrap strategy was flagged for.

**Decision required:** which CV source does `adf()` treat as authoritative for the
printed 1/5/10% table, and do we expose both MacKinnon (2010) values (backend)
and offer a Fuller (1976) path to match Stata's printed numbers? Recommended:
report the MacKinnon (1994) **p-value** as the primary parity anchor (matches
Stata), and document that the printed CVs are MacKinnon (2010). Lag default must
be set deliberately (see §5).

### 1.2 Phillips-Perron (`pp()`)

| Dimension | `arch.unitroot.PhillipsPerron` | Stata `pperron` | R `ur.pp` |
|---|---|---|---|
| Regression | y on yₜ₋₁ (+trend), **no aug lags** | y on yₜ₋₁ (+trend), levels | y on yₜ₋₁ (+trend) |
| Kernel | **Bartlett** (Newey-West, `cov_nw`) | **Bartlett** (NW) | **Bartlett** (NW) — *not* QS |
| **Default bandwidth** | `12·(n/100)^¼` (Schwert, fixed) | `int(4·(T/100)^(2/9))` | `trunc(4·(n/100)^0.25)` ("short") |
| Statistic default | **Z(t)** (`test_type="tau"`) | **Z(t)** (+ Z(ρ) also printed) | **Z(alpha)** (`type="Z-alpha"`) |
| **Critical VALUES** | MacKinnon **ADF** tables (adf-t / adf-z) | **Fuller (1976)** | inline PP 1/n response surface |
| p-value | MacKinnon (1994) | MacKinnon (1994) on Z(t) | none |

**Conflict:** bandwidth exponent differs (2/9 Stata vs 1/4 R/arch); R default
stat is Z-alpha not Z-tau; **CV vintage differs** (arch reuses MacKinnon ADF
tables; Stata uses Fuller 1976; R uses a PP response surface). Another
silent-mismatch surface.

### 1.3 KPSS (`kpss()`)

| Dimension | `arch.unitroot.KPSS` | R `ur.kpss` | Stata |
|---|---|---|---|
| Null | stationary | stationary | (community only) |
| Type default | `"c"` (level) | `"mu"` (level) | — |
| **CV vintage** | **Hobijn et al. (2004)** generalized (arch's own 100M-rep sim) | **Kwiatkowski et al. (1992)** original asymptotic (0.347/0.463/0.574/0.739 @10/5/2.5/1%) | — |
| **Default bandwidth** | Hobijn (1998) data-dependent (`n^(2/9)` rule) | `trunc(4·(n/100)^0.25)` | — |

**Conflict:** CV values are numerically close but from **different sources**
(Hobijn 2004 vs KPSS 1992). e.g. level 5%: arch ≈ 0.4614 vs KPSS 1992 = 0.463.
Small but real divergence; the printed 1% differs slightly more. Decision needed
on which to present as authoritative.

### 1.4 DFGLS (`dfgls()`)

| Dimension | `arch.unitroot.DFGLS` | Stata `dfgls` |
|---|---|---|
| GLS detrend | ERS, cbar = −7.0 (c) / −13.5 (ct) | ERS, cbar = −7.0 / −13.5 (matches) |
| **Default lags** | auto AIC, max `12·(n/100)^¼`, **Perron-Qu OLS-detrended lag select** | Schwert `int(12·(n/100)^0.25)`, Ng-Perron sequential-t / SIC / MAIC |
| **CV vintage** | MacKinnon `dfgls` tables (arch sim) | ERS (1996) / Fuller / Cheung-Lai (1995) |

**Conflict:** CV vintage differs (arch MacKinnon-dfgls vs Stata ERS1996/Fuller).
Lag-selection method differs (AIC vs Ng-Perron sequential-t). Closer than ADF but
still not identical at small N.

### 1.5 Zivot-Andrews (`zivot_andrews()`)

| Dimension | `arch.unitroot.ZivotAndrews` | R `ur.za` |
|---|---|---|
| Break models | "c" / "t" / "ct" | "intercept" / "trend" / "both" |
| **Default lags** | ADF(ct) AIC selection | **0** (`lag=NULL`→0) |
| **CV vintage** | **arch Monte Carlo** (100k reps, 2000 obs) | **Zivot-Andrews (1992)** asymptotic |
| p-value | none | none |

**Conflict:** CVs are from **different sources** (arch's own MC vs ZA1992
asymptotic) and can differ non-trivially. Default lag 0 (R) vs AIC (arch).

---

## 2. ARIMA / ARMA (`arima()` / `arma()`)

| Dimension | `statsmodels.ARIMA` | Stata `arima` | R `stats::arima` |
|---|---|---|---|
| **Default method** | **pure ML** (state-space Kalman, `method='statespace'`) | **pure ML** (Kalman, `ml model ... maximize`) | **CSS-ML** (`method="CSS-ML"` default) |
| Constant/mean | constant **in** if d=0; **dropped** if d>0/D>0 | **in** by default (`noconstant` to drop) | `include.mean=TRUE` |
| Dist / trend | `trend=None`→`'c'` (d=0) else `'n'` | constant by default | mean in |
| Optimizer | Newton/BFGS (statespace) | `bhhh 5 bfgs 10`, `vce opg` | `optim` BFGS, `transform.pars` |

**Conflict (real):** **R defaults to CSS-ML; statsmodels and Stata default to
pure ML.** To match R's `arima()` output, OE must either force a CSS-ML-equivalent
path or document that `arima()` follows Stata/statsmodels ML by default. Known
historical **MA/AR sign-convention** mismatches between statsmodels and
R/Stata also require an empirical coefficient check before parity is claimed.

**No GARCH/ARIMA variance-targeting issue** — see §3.

---

## 3. GARCH (`garch()`) — CLEAN, low-risk

| Dimension | `arch.arch_model` | Stata `arch` (+`garch()`) | R `rugarch` (documented) |
|---|---|---|---|
| Default volatility | **GARCH(1,1)** (`vol="GARCH", p=1, q=1`) | user must specify `arch()/garch()` | `garchOrder=c(1,1)` |
| Default mean | Constant (in) | constant in | `include.mean=TRUE` |
| **Default distribution** | **Normal** | **Gaussian** | **"norm"** |
| **Variance targeting** | **none — ω estimated freely (full MLE)** | **none — ω estimated freely** | **default FALSE — ω estimated freely** |
| Estimator | MLE, all params joint | MLE (BHHH/BFGS, `vce opg`) | MLE |

**Finding:** **No variance-targeting divergence.** arch (confirmed: zero
`variance_targeting` hits in source; ω is parameter[0]), Stata `arch`, and
rugarch all estimate the variance constant freely. All default to Gaussian. The
only divergence is the GARCH lag default (arch/rugarch = (1,1); Stata requires
explicit `garch(1)`), which is trivially reconciled by always passing the order.
**GARCH is therefore the lowest-risk v1.1.0 item and can ship first once the
unit-root CV decision (§1,§2) is settled.**

---

## 4. Required product decisions (gating wrapper code)

1. **ADF critical-value source.** Backend supplies MacKinnon (2010) values; Stata
   prints Fuller (1976); R prints banded Fuller. p-values agree (MacKinnon 1994).
   *Pick one authoritative CV source for the printed table, or expose both.*
2. **KPSS / DFGLS / ZA critical-value source.** Different simulation bases
   (Hobijn 2004 / MacKinnon-dfgls / arch-MC) vs (KPSS 1992 / ERS1996 / ZA1992).
   *Decide authoritative source per test.*
3. **Lag / bandwidth DEFAULTS.** No two tools agree for ADF (0 / AIC-auto /
   fixed-1) or PP/KPSS bandwidth (exponents 2/9 vs 1/4). *Decide OE defaults and
   whether to mirror Stata (0 lags) or arch (auto).*
4. **ARIMA estimation method default.** Match Stata/statsmodels (pure ML) or R
   (CSS-ML)? *Pick; document the other as an option.*
5. **R `forecast`/`rugarch`** are not installed here — their defaults above are
   documented, not source-verified. *Install and re-verify before claiming R
   parity for GARCH/ARIMA.*

## 5. Recommended wrapper posture (pending decision)

- Expose **every** relevant knob (`lags`, `trend`, `test_type`, `kernel`,
  `bandwidth`, `method`) so the user can dial in Stata/R-equivalent behavior
  rather than being locked to the backend default.
- Make `adf()`/`pp()` default to **Stata-equivalent** lag behavior where a
  reviewer expects Stata parity, but document the arch auto-lag alternative.
- Report **MacKinnon (1994) p-values** as the cross-check anchor (agrees with
  Stata) and label the CV table's vintage explicitly in `summary()`.

*Recon performed 2026-07-14. No source files under `open_econs/` were modified
by this recon except this document.*

---

## 6. v1.1.0 tolerance-standard re-audit (2026-07-15)

The v1.1.0 suite shipped under a looser tolerance standard. With rule 2 tightened
to a hard **1e-6** ceiling, every cross-tool assertion was re-opened and actually
attempted to be closed before being accepted. Findings below. Tightened assertions
were committed separately from the documented exceptions (rule 4).

### 6.1 Confirmed <= 1e-6, tightened (no exception needed)

| Anchor | Quantity | Observed relative gap | Action |
|--------|----------|----------------------|--------|
| ADF vs Stata | stat + MacKinnon p-value (c,ct) | 7e-9 / 1e-11 / 2e-8 | rtol 1e-4 -> 1e-6 |
| PP vs Stata | stat + p-value (c,ct) | 5e-9 / 8e-9 / 1e-8 | rtol 1e-4 -> 1e-6 |
| ADF vs R (ur.df) | stat | 1e-15 | rtol 1e-4 -> 1e-6 |
| KPSS vs R (ur.kpss, matched bandwidth) | stat | 6e-16 / 2e-15 | rtol 1e-4 -> 1e-6 |
| ZA vs R (ur.za) | stat | 9e-15 / 1e-14 | rtol 1e-4 -> 1e-6 |
| DFGLS (OE == arch identity) | stat | 1e-12 abs | unchanged (already <= 1e-6) |
| KPSS self-consistency guard | own default | 1e-9 | unchanged (already <= 1e-6) |

OE matches Stata on ADF/PP to <=8e-9 and matches R (urca) on ADF/KPSS/ZA to
floating-point precision. These are now asserted at the maximally-tight 1e-6.

### 6.2 PP-vs-R formula divergence (DOCUMENTED EXCEPTION, exceeds 1e-6)

OE vs R `ur.pp` Z-tau differs by **1.4e-5 (c) / 5.9e-6 (ct)** -- above 1e-6.

**Root cause (source-confirmed, not tolerable noise).** R's `ur.pp` Z-tau
statistic uses the **dependent-variable** (y_t) variance in the long-run
correction term, whereas `arch` (and Stata `pperron`) use the **regressor**
(y_{t-1}) variance -- the textbook Hamilton form. Verified by replicating both
formulas exactly:
- arch/Stata form: Z_tau = sqrt(gamma0/lam2)*((rho-1)/sigma) - 0.5*((lam2-gamma0)/lam)*(n*sigma/s)
  where n*sigma/s = n/sqrt(sum((y_l1 - mean)^2)) (regressor variance).
- R `ur.pp` form: ... - lambda.prime*sqrt(sig)/sqrt(myybar) where
  myybar = (1/n^2)*sum((y_t - mean(y_t))^2) (dependent-variable variance).

Replicating arch's exact formula reproduces OE/arch to 1e-12; replicating R's
exact formula reproduces the R fixture to 1e-15. The two are mathematically
distinct estimators. OE and Stata agree to 8e-9, so OE is on the textbook side;
R's `ur.pp` is a known variant. The PP-vs-R assertions therefore use
PP_R_RTOL = 2e-5 (genuine envelope + margin) -- an intentional, evidenced
exception to the 1e-6 ceiling.

### 6.3 ARIMA flat-likelihood exception (DOCUMENTED EXCEPTION, exceeds 1e-6)

Log-likelihood agrees across OE / Stata / R to **~2e-9** (tightened to 1e-6).
Coefficients do NOT: AR1 rel 3.7e-5 (vs Stata) / 2.3e-5 (vs R); MA1 rel 6.2e-5 /
3.7e-5; const abs ~1.8e-5 (Stata) / 1.1e-5 (R). All above 1e-6.

**Root cause (demonstrated, not "optimizer noise").** The ARMA(1,1) likelihood
is flat in the AR/MA subspace near the optimum. Evaluated via statsmodels
mod.loglike at the Stata/R coefficient points: shifting the coefficients by the
observed ~4e-5 changes the log-likelihood by only **2.3e-9** (gradient ~6e-5 per
unit). Tightening statsmodels convergence (maxiter up to 1e5, tol=1e-14, lbfgs)
does NOT move OE's point at all -- it is already at its convergence floor; Stata
and R are at *their* distinct floors (they even differ from each other by ~1e-5
on ar1). The coefficient is simply not identified to 1e-6 by the likelihood; the
LL -- the genuine MLE invariant -- is the tight cross-tool quantity and it
matches to 2e-9. Coefficient assertions use RTOL_COEF = 1e-4, const uses
ATOL_CONST = 1e-4 (genuine envelope + margin) -- intentional, evidenced
exceptions.

### 6.4 GARCH omega-beta ridge exception (DOCUMENTED EXCEPTION, exceeds 1e-6)

Cross-tool coefficient spread ~1-1.5% (beta rel 1.5e-2 vs Stata / 1.0e-2 vs R);
reported LL spread ~1.4e-4 relative. Both above 1e-6.

**Root cause (demonstrated, NOT the prior "optimizer noise" claim).** The prior
v1.1.0 handoff attributed the 2e-2 gap to "optimizer noise across three
independent MLE implementations." This is REFUTED by evidence: `arch` is
**deterministic to ~1e-7** across different starting values and tight tolerances
(options={'maxiter':20000,'ftol':1e-14}), so the gap is not arch scattering. The
true cause is the GARCH(1,1) **omega-beta ridge**: omega and beta are
near-collinear in h_t = omega + alpha*e^2 + beta*h_{t-1}, so the likelihood is
flat along the ridge. Evaluated via a manual Gaussian GARCH LL (unconditional
backcast) at the three committed param sets:
- arch LL = -826.4805, Stata LL = -826.4806 (1.1e-7 rel), R LL = -826.4823
  (2.1e-6 rel). All three are on the same likelihood ridge.
- Perturbing arch's optimum along the ridge (omega -1%, beta +compensating)
  changes the LL by only 1.6e-6 relative.

A secondary contributor is the presample/backcast variance initialization, which
shifts the *reported* LL by ~1.4e-4 relative between `arch` and Stata/R. The 2e-2
relative tolerance is the genuine cross-tool envelope with margin; it is an
intentional, evidenced exception to the 1e-6 ceiling and is flagged to the
project lead.

**Flag to lead:** three evidenced exceptions to the 1e-6 ceiling exist
(6.2 PP-vs-R, 6.3 ARIMA coeffs/const, 6.4 GARCH coeffs/LL). Each is root-caused
to a specific, demonstrated source-level cause (a formula-level convention in R's
ur.pp; a flat likelihood in ARIMA; the omega-beta ridge + presample init in
GARCH) -- none is an unexplained tolerance relax. The genuine MLE invariant
(log-likelihood) is <=2e-9 for ARIMA and ~2e-6 for GARCH; only the *coefficients*
diverge, because the likelihood does not identify them that tightly.

*Audit performed 2026-07-15. Only test tolerances and this document were modified;
no `open_econs/` source was changed.*
