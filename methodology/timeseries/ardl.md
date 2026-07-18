# ARDL, UECM & the Pesaran-Shin-Smith Bounds Test in Python — Stata `ardl` / R `ARDL` Parity

`open-econs` implements the **autoregressive distributed-lag (ARDL)** model, its
**unrestricted error-correction (UECM / ECM)** reparameterization, and the
**Pesaran-Shin-Smith (2001) bounds test for a level relationship**, wrapping
`statsmodels.tsa.ardl` and reconciled to Stata's SSC `ardl` (Kripfganz &
Schneider 2018) and R's `ARDL` / `dynamac` packages. This note records the
mathematics, the cross-tool convention crosswalk, and the runnable commands, so
future sessions do not re-derive them (standing rules 13, 16).

---

## 1. The model

### ARDL(p, q₁, …, q_k)

For a dependent series `y_t` and `k` regressors `x_{1t}, …, x_{kt}`:

```
y_t = c + Σ_{i=1}^{p} φ_i y_{t-i} + Σ_{j=1}^{k} Σ_{l=0}^{q_j} β_{j,l} x_{j,t-l} + ε_t
```

`p` is the autoregressive order; `q_j` the distributed-lag order of regressor
`j`. Deterministics (constant, trend) are governed by the PSS *case*
(Section 3).

### UECM (unrestricted error-correction form)

The ARDL is algebraically re-parameterized into the conditional ECM:

```
Δy_t = c + ρ y_{t-1} + Σ_{j=1}^{k} θ_j x_{j,t-1}
         + Σ_{i=1}^{p-1} ψ_i Δy_{t-i} + Σ_{j=1}^{k} Σ_{l=0}^{q_j-1} δ_{j,l} Δx_{j,t-l} + ε_t
```

- `ρ` (coefficient on the **level** `y_{t-1}`) is the **speed-of-adjustment /
  error-correction term**; `ρ = φ(1) − 1 = Σφ_i − 1`. Mean reversion ⇒ `ρ < 0`.
- `θ_j` are the level coefficients on `x_{j,t-1}`.
- **Long-run coefficient** on `x_j`:  `LR_j = −θ_j / ρ`.

---

## 2. The PSS (2001) bounds test

Tests `H0: ρ = 0 and θ_1 = … = θ_k = 0` (no level relationship) against the
alternative, using an **F-type** (Wald / #restrictions) statistic whose
asymptotic distribution is **non-standard** and bounded by two limiting cases:
all regressors I(0) (lower bound) and all I(1) (upper bound).

```
F = (1 / m) · R β̂ [ R V̂ R' ]^{-1} (R β̂)'
```

where `R` selects the `m` level coefficients tested in the given case
(`m = k + 1` for the pure cases, `k + 2` when a restricted deterministic is
also tested). This F definition is **identical across Stata `ardl`, R `ARDL`,
R `dynamac`, and statsmodels** — it is the one cross-tool ≤1e-6 agreement point.

**Fixture footgun (rule 18) — Stata `import delimited` reads `float`.** When the
Stata parity fixture is generated, `import delimited` reads numeric columns as
**single-precision (`float`) by default**. On this example (R²=0.988, near-
collinear lags) that input truncation shifts the OLS level coefficients by ~3e-6
and `e(F_pss)` by ~4e-5, producing a *spurious* Stata-vs-(R/statsmodels)
divergence that looks like a convention difference but is not — R `read.csv` and
pandas `read_csv` both default to double, so only the Stata leg was affected. The
fix (verified to restore <1e-6 parity on F/t/EC/LR) is `set type double` **before**
`import delimited` in `tests/stata/generate-fixtures/ardl.do`. This was diagnosed
by refitting the identical level regression with numpy QR *and* normal-equations
(both = statsmodels = R to 1e-14) while Stata-default alone disagreed; adding
`set type double` collapsed the gap to 1e-13. Do not "fix" a future recurrence by
loosening the tolerance.

A companion **t-bounds test** on `ρ` (the `y_{t-1}` coefficient) exists in
Stata / R `ARDL` / `dynamac` but **NOT in statsmodels' `bounds_test`** — OE
computes it directly (`t = ρ̂ / se(ρ̂)`) to reach parity.

---

## 3. Case-numbering crosswalk (source-verified, all identical)

All four tools use the **same PSS Table CI numbering** — no remap needed:

| Case | Stata `ardl`           | R `ARDL` | statsmodels `case=` | Deterministic restriction              |
|------|------------------------|----------|---------------------|----------------------------------------|
| 1    | `noconstant`           | `n`/1    | 1 (`trend="n"`)     | no constant, no trend                  |
| 2    | constant *restricted*  | `rc`/2   | 2 (`trend="c"`, tested) | restricted constant, no trend      |
| 3    | `constant`             | `uc`/3   | 3 (`trend="c"`)     | unrestricted constant, no trend        |
| 4    | constant + `trendvar` *restricted* | `ucrt`/4 | 4 (`trend="ct"`, trend tested) | unrestr. constant, restricted trend |
| 5    | constant + `trendvar`  | `ucut`/5 | 5 (`trend="ct"`)    | unrestricted constant, unrestricted trend |

Sources: Stata `ardl.ado` L127–139 / help L317–327; R `ARDL` `parse_case`;
statsmodels `UECMResults.bounds_test` L2381–2386.

**t-bounds case folding (footgun, rule 18):** for the *t*-statistic only, Stata
silently folds case 2→3 and 4→5 (`ardlbounds.ado` L30 — "t-statistic unaffected
by restrictions on deterministic components"), while R `ARDL` **forbids** cases
2/4 for the t-test entirely. OE follows Stata's fold for the t-bounds and
documents it; F-bounds is never folded.

---

## 4. Critical-value vintages (the `cv_vintage` toggle — rule 15)

Only **PSS(2001) asymptotic** tables are common to all four tools; that is the
OE default and the only vintage asserted at ≤1e-6 cross-tool.

| Tool                    | Default CV vintage                              | Notes |
|-------------------------|-------------------------------------------------|-------|
| Stata `ardl` / `estat ectest` | Kripfganz-Schneider (2020) response surface | `asymptotic` option; legacy `estat btest` → PSS2001 / Narayan2005 |
| Stata `ardlbounds, table` | PSS2001 (n≥83) else Narayan2005              | auto by sample size |
| R `ARDL`                | PSS2001 asymptotic (`exact=FALSE`, k≤10)        | `exact=TRUE` → simulate at T=n |
| R `dynamac` (`pssbounds`) | Narayan2005 small-sample; PSS asymptotic for n>80 | no toggle |
| statsmodels             | PSS2001 asymptotic (simulated, k≤10)            | `asymptotic=False` → resimulate at T |

OE `cv_vintage`:
- `"pss2001"` **(default)** — the **published asymptotic PSS(2001) Table CI
  (F-bounds) and CII (t-bounds)** I(0)/I(1) critical values, cases 1-5, k=0-10,
  α ∈ {0.10, 0.05, 0.025, 0.01}. **These are served from an OE-embedded table,
  NOT from statsmodels' Monte-Carlo simulation** (see root-cause note below).
  Asserted ≤1e-6 vs Stata `ardlbounds` (n≥83), R `ARDL` (`exact=FALSE`), and R
  `dynamac::pssbounds` (n>80).
- `"statsmodels"` — statsmodels' **simulated** finite-sample F-bounds
  (`asymptotic=False`, re-simulated at the sample size). This is a documented
  divergence, NOT a cross-tool parity anchor.

**Root cause / footgun (rule 16, 18).** statsmodels' `UECMResults.bounds_test`
with `asymptotic=True` does **not** return the published PSS(2001) table — it
returns Monte-Carlo *simulated* asymptotic critical values (via
`statsmodels.tsa.ardl._pss_critical_value`). For the canonical denmark case
(case 3, k=3, α=1%) statsmodels gives an I(1) upper bound of ≈6.32, whereas the
published PSS table (and R/Stata) give **5.61** — a ~0.7 gap, far outside 1e-6.
OE therefore embeds the published table (extracted verbatim from
`ARDL:::crit_val_bounds_pss2001`, which reproduces Pesaran-Shin-Smith 2001
Tables CI/CII) and only uses the F-**statistic** from statsmodels (which *is*
convention-free and matches to 1e-6). The `asymptotic` flag only ever affected
statsmodels' CV table, never the statistic.

**P-values do NOT match cross-tool** even when CVs agree: statsmodels uses a
log-polynomial response surface, Stata uses MacKinnon(1996) auxiliary
regressions, R `ARDL` interpolates its simulated CDF. Assert **CVs and the
F/t-statistic**, not p-values, across tools; assert p-values only within a
single tool.

---

## 5. Long-run / EC sign convention (footgun, rule 18)

statsmodels `ci_params` reports the long-run coefficient as `+θ_j / ρ` (it
divides level coefs by the `y.L1` coef with **no leading minus** and normalizes
the `y.L1` entry to 1.0). Stata `ardl, ec` and R `ARDL::multipliers()` report
`−θ_j / ρ` — the **opposite sign**, and do not include the normalized `1.0`.

OE default (`lr_sign="stata"`): report `LR_j = −θ_j / ρ`, matching Stata/R,
dropping the `y.L1=1` base. `lr_sign="statsmodels"` returns the raw
`ci_params`. The **speed-of-adjustment** coefficient `ρ` itself (raw `y_{t-1}`
regression coefficient) is signed identically across all three tools.

---

## 6. Default IC / max lags (pin explicitly)

| Tool                       | Default IC | Default max lag |
|----------------------------|------------|-----------------|
| Stata `ardl`               | BIC        | 4               |
| R `ARDL::auto_ardl`        | AIC        | (mandatory)     |
| statsmodels `ardl_select_order` | BIC   | (mandatory)     |

Stata & statsmodels default BIC; R `ARDL` defaults AIC. Automated
lag-selection parity requires pinning `ic=` explicitly. For a **fixed** lag
order (what the bounds test needs) IC is irrelevant.

---

## 7. Usage (Stata-manual style)

```python
import open_econs as oe

# fixed-order ARDL(1, 1) of y on x, unrestricted constant (case 3)
res = oe.ardl_fit(df, "y", exog=["x"], order=(1, 1), trend="c")
res.summary()

# error-correction form + long-run coefficients (Stata `ardl, ec`)
u = oe.uecm_fit(df, "y", exog=["x"], order=(1, 1), trend="c")
u.long_run          # −θ/ρ, Stata/R sign
u.ec_term           # ρ, speed of adjustment

# PSS bounds test for a level relationship
bt = u.bounds_test(case=3)          # cv_vintage="pss2001" default (published table)
bt.f_stat, bt.f_crit_upper          # F statistic + I(0)/I(1) bounds
bt.t_stat, bt.t_crit_upper          # t-bounds (OE-computed)

# statsmodels' simulated finite-sample bounds (documented divergence)
bt_sim = u.bounds_test(case=3, cv_vintage="statsmodels")

# automatic order selection (pin IC for cross-tool parity)
sel = oe.ardl_select_order(df, "y", exog=["x"], maxlag=4, maxorder=4, ic="bic")

# via a TimeSeriesContext
ctx = oe.TimeSeriesContext(df)
ctx.ardl_fit("y", exog=["x"], order=(1, 1))
```

Stata equivalent:

```stata
ssc install ardl
ardl y x, lags(1 1) ec
estat ectest
```

R equivalent:

```r
library(ARDL)
m  <- ardl(y ~ x, data = df, order = c(1, 1))
bt <- bounds_f_test(m, case = 3)
mult <- multipliers(m)          # long-run coefficients
```

---

## 8. Parity summary

- **Assert ≤1e-6 cross-tool:** ARDL/UECM coefficients & SEs, the bounds
  **F-statistic**, the **t-statistic** (Stata / R `ARDL`), and the **PSS2001
  asymptotic I(0)/I(1) critical values**.
- **Toggle + cover both (rule 15):** `cv_vintage` (pss2001 published table vs
  statsmodels simulated), `lr_sign` (stata vs statsmodels).
- **Do NOT assert cross-tool:** p-values (method-specific), statsmodels'
  simulated finite-sample CVs, automated IC selection unless `ic=` is pinned.

*Recon source-verified 2026-07-18 against Stata `ardl.ado`/`ardlbounds.ado`,
R `ARDL`/`dynamac` sources, and statsmodels `tsa/ardl/model.py`.*
