# VAR / VECM / Johansen Backend Recon (`statsmodels.tsa.vector_ar` vs Stata / R)

**Status:** Source-verified recon, pre-build. Written during source recon for
v1.1.1. **No wrapper code has been written yet.** This document records the
convention gaps discovered between the wrapped backend
(`statsmodels.tsa.vector_ar` — `VAR`, `VECM`, `coint_johansen`,
`select_order`) and the Stata/R reference tools, so the parity strategy can be
decided **before** any code is committed.

**Standing-rule reminder:** wrapping is an implementation strategy, not a parity
exemption. Every gap below must be *either* root-caused-and-fixed,
corrected-for in the OE wrapper, or documented as a source-confirmed intentional
convention. None may be silently absorbed.

**Tools verified (all present and read at source level):**
- Stata 17 base: `var.ado` (v1.10.1), `vecrank.ado` (v1.1.4), `vargranger.ado`
  (v1.6.3), `_vecgetcv.ado` (v1.0.1), `_vecgtn.ado` (v1.0.1), `_vecu.ado`.
- R 4.6.1: `vars` (VAR / `vec2var` / `VARselect` / `causality`) and `urca`
  (`ca.jo`) installed. Source read via `Rscript` (functions `VARselect`,
  `causality`, `ca.jo` printed; hard-coded CV tables inspected).
- `statsmodels` 0.14.6: `tsa/vector_ar/var_model.py` (class `VAR`,
  `select_order`, `test_causality`, `info_criteria`), `tsa/vector_ar/vecm.py`
  (`VECM`, `coint_johansen`, `select_order`), `tsa/coint_tables.py`
  (`c_sja`, `c_sjt`).

---

## 1. VAR lag-order selection criteria (AIC / BIC / HQIC / FPE)

### 1.1 Exact formulas read from source

**Stata `var`** (`var.ado`):
- FPE (always, lines 290): `fpe = ((T+eqparm)/(T-eqparm))^K * detsig_ml`,
  where `eqparm = e(df_eq)` = params *per equation* (AR + const + exog),
  `K = e(neqs)`, `detsig_ml` = ML determinant of residuals, `T = e(N)`.
- Default AIC/BIC/HQIC (lines 297-299): `aic = -2*(ll/T) + (2*tparms)/T`,
  `hqic = -2*(ll/T) + (2*ln(ln(T))/T)*tparms`,
  `sbic = -2*(ll/T) + (ln(T)/T)*tparms`, `tparms = e(tparms)` = total params.
- `lutstats` option (lines 302-304): same log-det form but the penalty uses
  `arparms = K^2*maxlag` **only** (AR params; deterministic/exog terms
  EXCLUDED): `aic = ln(detsig_ml) + (2*arparms)/T`, etc.

**R `vars::VARselect`** (source, `criteria[1..4]`): `sigma.det = det(crossprod(resids)/sample)`
(ML det, ÷ `sample = T - lag.max`, fixed across lags), `nstar = i*K^2 + K*detint`
(per-equation param count, `detint` = deterministic columns):
- `AIC(n) = log(sigma.det) + (2/sample)*(i*K^2 + K*detint)`
- `HQ(n)  = log(sigma.det) + (2*log(log(sample))/sample)*(i*K^2 + K*detint)`
- `SC(n)  = log(sigma.det) + (log(sample)/sample)*(i*K^2 + K*detint)`
- `FPE(n) = ((sample + nstar)/(sample - nstar))^K * sigma.det`

**statsmodels `VAR.info_criteria`** (`var_model.py:2282-2303`, `nobs`,
`df_model = neqs*k_ar + k_exog`, `ld = logdet(Sigma_mle)`):
- `aic  = ld + (2/nobs)*free_params`
- `bic  = ld + (log(nobs)/nobs)*free_params`
- `hqic = ld + (2*log(log(nobs))/nobs)*free_params`
- `fpe  = ((nobs + df_model)/df_resid)^neqs * exp(ld)`
  (`free_params = k_ar*neqs^2 + neqs*k_exog`; `df_resid = nobs - df_model`).

### 1.2 Verdicts

- **FPE — NO-CONFLICT (identical).** All three use the Lütkepohl form
  `((T + m)/(T - m))^K * det(Σ_ml)` where `m` = parameters *per equation*
  (AR + deterministic + exog) and `Σ_ml` is the biased (÷T) ML covariance.
  R `vars` and statsmodels are term-for-term identical (`m = nstar = df_model`);
  Stata's `eqparm` is the per-equation count too, so `((T+eqparm)/(T-eqparm))^K`
  matches. Denominator convention is `T - m` (not `T`, not `T - m - 1`) in all
  three. No small-sample adjustment (`T` rather than `T - m`) anywhere — this is
  the textbook Lütkepohl convention, not the `arch`/AICc-style correction.
- **AIC/BIC/HQIC (R vs statsmodels) — NO-CONFLICT (identical).** R `vars` and
  statsmodels use the *same* `ln-det + penalty/nobs` representation with the
  *same* `m` (deterministic terms INCLUDED in the penalty). Values match.
- **AIC/BIC/HQIC (Stata default vs R/statsmodels) — NO-CONFLICT on argmin,
  CONFLICT on reported absolute value.** Stata's default uses the
  `-2*ll/T + penalty/T` representation. Up to the additive constant
  `K*ln(2π) + K` (and the `ln det Σ` vs `-2ll/T` identity) it selects the same
  lag order, but the printed numbers differ from R/statsmodels. This is a
  *display* conflict, not a selection conflict.
- **Stata `lutstats` option — REAL CONFLICT.** When `lutstats` is set, Stata's
  AIC/HQIC/SBIC penalty uses `arparms = K^2*lag` and **drops the deterministic/
  exog terms from the penalty**, whereas R `vars` and statsmodels **include**
  them. So Stata `lutstats`-flavoured IC can select a *different* lag order than
  the R/statsmodels default. Flagged as a decision (see §5-1).

---

## 2. Johansen trace / max-eigenvalue critical-value table provenance

### 2.1 Source of each tool's CVs (read from source)

- **statsmodels** (`vecm.py:603-737`, `coint_johansen`): for each rank `i` it
  sets `cvm[i,:] = c_sja(neqs-i, det_order)` and
  `cvt[i,:] = c_sjt(neqs-i, det_order)` (`vecm.py:733-734`). These come from
  `statsmodels.tsa.coint_tables` (`c_sja`/`c_sjt`), whose own docstring
  (`coint_tables.py:1-36, 104-136`) states verbatim: *"The values returned by
  the function were generated using a method described in MacKinnon (1996),
  using his FORTRAN program johdist.f"* and cites **MacKinnon, Haug, Michelis
  (1996), 'Numerical distribution functions of likelihood ratio tests for
  cointegration'**. These are response-surface values, valid for `neqs ≤ 12`,
  `det_order ∈ {-1,0,1}`.
- **Stata `vecrank`** (`_vecgetcv.ado:9-19`): comment reads *"critical values
  from Osterwald-Lenum Oxford Bulletin of Economics and Statistics 54(3) 1992
  pp 461-472"*. The table has **5 columns** (none / rconstant / constant /
  rtrend / trend — see `_vecgtn.ado`), i.e. the finite-sample **Osterwald-Lenum
  (1992)** tables, no response surface.
- **R `urca::ca.jo`** (`ca.jo` source): the `cval` slot is filled from
  **hard-coded arrays** `cv.none`, `cv.const`, `cv.trend` (dim `c(11,3,2)`:
  11 ranks × {10%,5%,1%} × {max-eigen, trace}). These arrays are the
  **Osterwald-Lenum (1992)** finite-sample numbers (e.g. `cv.const` trace r=0
  5% = 19.96; `cv.none` trace r=0 5% = 17.95; these are the canonical
  Osterwald-Lenum values, not a MacKinnon response surface).

### 2.2 Empirical comparison (2-variable system, r=0, 5% point, trace test)

| Specification | statsmodels `coint_johansen` (MacKinnon 1996) | R `urca` (Osterwald-Lenum 1992) |
|---|---|---|
| `det_order=-1` / `ecdet="none"` | **12.321** | 17.95 |
| `det_order=0` / `ecdet="const"` | **15.494** | 19.96 |
| `det_order=1` / `ecdet="trend"` | **18.398** | 25.32 |

Stata `vecrank` columns for the same cases (K-r=2, 5%): none = 12.53,
rconstant = 19.96, constant = 15.41, rtrend = 25.32, trend = 18.17 — i.e.
**Stata == urca == Osterwald-Lenum (1992)**; statsmodels sits on a **different
(MacKinnon 1996) surface** and is materially lower (e.g. constant case 15.49 vs
19.96, a ~4.5-point gap on a statistic of ~20-30).

### 2.3 Verdict

- **CRITICAL CONFLICT — CV table provenance.** The OE backend
  (`statsmodels`) draws Johansen trace/max-eigen critical values from the
  **MacKinnon-Haug-Michelis (1996) response surface**, while **both Stata
  (`vecrank`) and R (`urca::ca.jo`) use the finite-sample Osterwald-Lenum (1992)
  tables**. The two sources are not numerically equivalent; the gaps above are
  typical and will frequently flip the selected cointegration rank. This is the
  single most important finding of this recon and must be surfaced to the lead
  (decision §5-2). It is *not* a pick-one-and-document case in the sense of
  "cosmetic": different CVs ⇒ different rank ⇒ different model.
- NO-CONFLICT on the *statistic* itself: all three compute the same
  trace (`-T·Σ ln(1-λᵢ)`) and max-eigen (`-T·ln(1-λᵢ)`) statistics and the same
  eigenvalue ordering. Only the comparison CVs diverge.
- NO-CONFLICT on `det_order` range limits: statsmodels supports `det_order ∈
  {-1,0,1}` and `neqs ≤ 12`; urca warns above 11 vars and Stata above 12 — all
  three bound system size similarly.

---

## 3. Deterministic-term / cointegration-restriction case conventions

### 3.1 3-way mapping (by Johansen/Lütkepohl specification meaning)

| Deterministic spec (Lütkepohl) | Stata `vec`/`vecrank` `trend()` | R `urca::ca.jo` (`ecdet`/`spec`) | statsmodels `coint_johansen` | statsmodels `VECM` |
|---|---|---|---|---|
| none | `none` | `ecdet="none"` | `det_order=-1` | `det_order=-1` |
| constant **in** CE (restricted) | `rconstant` (rc) | `ecdet="const"`, `spec="longrun"` | `det_order=0` | `deterministic="ci"` / `det_order=0` |
| constant **outside** CE (unrestricted) | `constant` (c) | `ecdet="const"`, `spec="transitory"` | **not available** | `det_order=-1` + `exog=ones` |
| const+trend **in** CE (restricted) | `rtrend` (rct) | `ecdet="trend"`, `spec="longrun"` | `det_order=1` | `deterministic="li"` / `det_order=1` |
| const+trend **outside** (unrestricted) | `trend` (ct) | `ecdet="trend"`, `spec="transitory"` | **not available** | `det_order=-1` + `exog=ones+trend` |

Notes from source:
- Stata column→trend mapping verified in `_vecgtn.ado` (none=1, rconstant=2,
  constant=3, rtrend=4, trend=5) and the 5-column CV matrix in `_vecgetcv.ado`.
- `urca::ca.jo` only exposes three `ecdet` values and two `spec` values; its
  hard-coded `cv.*` tables correspond to the *restricted* cases (`ecdet="const"`
  ≈ Stata `rconstant`, `ecdet="trend"` ≈ Stata `rtrend`). `spec="transitory"`
  changes the *model* (where the constant/trend enters) but **does not change the
  printed `cval`** — urca reuses the same Osterwald-Lenum table regardless of
  `spec`.
- `coint_johansen` only supports `det_order ∈ {-1,0,1}` and **always places the
  deterministic inside the cointegrating relation** (it detrends the data by the
  deterministic before testing — `vecm.py:679-692`). The unrestricted cases are
  reachable only through `VECM(..., exog=, exog_coint=)` (`vecm.py:39-85`), which
  `coint_johansen` does **not** expose.

### 3.2 Verdicts

- **CONFLICT (capability) — unrestricted-deterministic cases have no clean
  `coint_johansen` equivalent.** Stata `c`/`ct` and R `urca` `spec="transitory"`
  (constant/trend *outside* the CE relation) cannot be produced by
  `coint_johansen`; they require `VECM` with `exog`. OE must decide whether
  `johansen()` wraps `coint_johansen` (restricted-only, 3 cases) or `VECM`
  (full 5-case coverage) — decision §5-3.
- **CONFLICT (double) — even matched specs use different CV sources.** As shown
  in §2, the *same* semantic case (e.g. restricted constant) yields different CVs
  because statsmodels uses MacKinnon (1996) and Stata/urca use Osterwald-Lenum
  (1992). So the mapping table above is about *specification*, not about
  *numerical agreement*.
- **Caution on `det_order` semantics.** Empirically statsmodels `det_order=0`
  (MacKinnon surface, trace r=0 5% = 15.49) lands numerically *between* Stata's
  `constant` (15.41) and `rconstant` (19.96) columns, so the exact
  Osterwald-Lenum case it "should" align to is ambiguous. The semantic mapping
  (restricted constant) is the intended one, but do not assume numeric equality
  with any Stata/urca column. Flagged to lead (§5-3).

---

## 4. Granger causality test-statistic conventions

### 4.1 Source read

- **Stata `vargranger`** (`vargranger.ado`): if `small` is **not** set
  (default), it runs `qui test` and reports `r(chi2), r(df), r(p)` — an
  **asymptotic Wald χ²** test (lines 113-114). With `small`, it reports
  `r(F), r(df), r(df_r), r(p)` — a **small-sample F** test (lines 132-134). The
  denominator df is the estimator's residual df.
- **R `vars::causality`** (`causality` source): operates on a `varest` object
  (for a VECM you must first `vec2var()`). Default returns an **F-Test**
  (`STATISTIC` named "F-Test"), `df1 = p*length(y1)*length(y2)` (num
  restrictions), `df2 = K*obs - length(PI)` (residual df), p-value from
  `pf(..., df1, df2)`. It *also* returns a separate **instantaneous-causality**
  χ² test (`result2`, "Chi-squared", df = N). So R's **default is the small-sample
  F**, with an extra instantaneous-causality χ² test that Stata/`vargranger` and
  statsmodels do not produce.
- **statsmodels `VAR.test_causality`** (`var_model.py:1903-2016`): `kind="f"`
  (default) → `statistic = lam_wald/num_restr`, `df = (num_restr, K*df_resid)`,
  F-distributed (`var_model.py:2011-2014`). `kind="wald"` → χ² with `df =
  num_restr` (`var_model.py:2008-2010`). `num_restr = len(causing)*len(caused)*p`.

### 4.2 Verdicts

- **CONFLICT (default statistic type).** Stata `vargranger` defaults to
  **asymptotic χ²**; **R `vars` and statsmodels both default to small-sample
  F**. To match Stata, OE must default `granger()` to `kind="wald"` (χ²); to
  match R, leave it at F. This is a genuine default divergence — decision §5-4.
- **NO-CONFLICT (F denominator df, R vs statsmodels).** Both use `df2 = K ·
  residual_df` (R: `K*obs - length(PI)`; statsmodels: `K*df_resid`), and
  `df1 = p·(#causing)·(#caused)`. Numerically identical for the same data/lag.
- **CONFLICT (instantaneous causality).** R `vars::causality` additionally
  reports an instantaneous-causality χ² test; statsmodels `test_causality` does
  **not** (Granger only), and Stata `vargranger` does not either. If OE wants
  R-parity it must add this; otherwise document it as R-only. Decision §5-4.
- **CAPABILITY GAP (VECM Granger).** statsmodels has **no** `test_causality` on
  `VECM` (only on `VAR`); R reaches VECM Granger via `vec2var()` → `causality`.
  OE must decide whether `granger()` on a VECM converts via `vec2var` (mirroring
  R) or is unsupported — decision §5-4.

---

## 5. Required product decisions (gating wrapper code)

1. **VAR IC `lutstats` parity (§1).** Stata `lutstats` excludes deterministic
   terms from the AIC/HQIC/SBIC penalty; R `vars`/statsmodels include them.
   *Should OE's `var()` expose a `lutstats`-style IC (match Stata's alternate)
   or stick to the R/statsmodels default (deterministic included)?* Recommended
   default: match R/statsmodels (include deterministic), since that is what the
   backend computes natively; expose `lutstats` only if Stata parity is
   required. (Absolute-value display still won't equal Stata's `-2ll/T` form —
   decide whether to re-center or just document.)
2. **Johansen CV table source (§2 — most important).** Backend = MacKinnon (1996)
   response surface; Stata & R = Osterwald-Lenum (1992). *Which table is
   authoritative for OE's `johansen()` printed 5%/1% critical values and rank
   selection?* This changes rank selection, so it is not cosmetic. Recommended
   default: match **Stata** (Osterwald-Lenum 1992) for reviewer expectations,
   which means OE must recompute CVs (or wrap `urca`'s tables / hard-code
   Osterwald-Lenum) rather than trust `coint_johansen`'s native values — OR
   document that OE reports MacKinnon (1996) and differs from Stata. Leave the
   call to the lead.
3. **Deterministic-case coverage (§3).** `coint_johansen` covers only the 3
   *restricted* cases (`det_order ∈ {-1,0,1}`); Stata `c`/`ct` and R
   `spec="transitory"` (unrestricted) need `VECM`+`exog`. *Should OE's
   `johansen()` wrap `coint_johansen` (restricted-only) or `VECM` (full 5-case
   coverage, matching Stata/urca), and how should the ambiguous `det_order`
   numeric alignment be labelled?* Recommended: wrap `VECM` for full coverage and
   label which Osterwald-Lenum column each case maps to.
4. **Granger default statistic + VECM (§4).** Stata defaults to χ², R/statsmodels
   to F. *Should OE's `granger()` default to F (R/statsmodels) or χ² (Stata)?*
   And should VECM Granger be supported via `vec2var` (R-style), and should the
   instantaneous-causality χ² test be added? Recommended default: F (matches R
   and the backend); add `chi2`/`small` switches to reach Stata; support VECM via
   `vec2var`. Leave final default to the lead.

## 6. Recommended wrapper posture (pending decision)

- **Do not trust `coint_johansen`'s native CVs for Stata/R parity.** They are
  MacKinnon (1996), not Osterwald-Lenum (1992). Either override the CV table or
  clearly label the statistic's provenance in `summary()`.
- **Expose every knob**: `ic` flavor (default vs `lutstats`), `crit` table
  source, `trend`/`det_order`/`exog`/`exog_coint`, and Granger `kind`
  (`f`/`wald`) + `small`. Let the user dial in Stata- or R-equivalent behavior
  rather than being locked to the backend default.
- **Report the statistic value as the cross-tool anchor** (trace / max-eigen /
  F / χ² all agree across tools to floating-point precision); the divergence is
  entirely in the *critical values* (Johansen) and the *IC representation*
  (VAR), not in the test statistics themselves.
- **Capability checklist before shipping**: (a) unrestricted-deterministic
  Johansen cases, (b) VECM Granger via `vec2var`, (c) instantaneous causality —
  none exist in `coint_johansen`/`VAR` today and must be built or documented as
  out-of-scope.

*Recon performed 2026-07-15. No source files under `open_econs/` were modified
by this recon except this document.*
