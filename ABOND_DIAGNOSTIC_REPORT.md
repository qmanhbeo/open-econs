# Arellano-Bond GMM Diagnostic Report v2: Collapsed-vs-Collapsed Comparison

## Executive Summary

The collapsed-vs-collapsed comparison **partially resolves** the coefficient mismatch but **does not resolve** the SE magnitude issue. The constant instrument was a red herring — confirmed dead end. Two independent issues remain:

1. **GMM instrument structure mismatch**: Stata's `collapse` averages across depths (1 GMM column total), while oe keeps one column per depth (3 GMM columns). This causes a 5-6% coefficient divergence and instrument count discrepancy (4 vs 5).
2. **SE magnitude (24x too small)**: Confirmed NOT a collapsed-vs-collapsed issue. The sandwich VCV computation itself produces SEs 24x smaller than Stata. Root cause is in the GMM weighting matrix or the sandwich formula, not the instrument structure.

---

## 1. Stata Baseline: Non-Collapsed (Original, no `collapse` option)

From `abond_diag.log`:

| Run | Lag Spec | Step | Instruments | b_L1.y | b_x | b_z | se_L1.y | se_x | se_z |
|-----|----------|------|-------------|--------|-----|-----|---------|------|------|
| A | lag(2 2) | 1-step | 4 | -0.066542 | 1.161509 | -0.310641 | 0.256343 | 0.184606 | 0.107294 |
| B | lag(2 4) | 1-step | 5 | -0.086714 | 1.147234 | -0.303538 | 0.245213 | 0.176805 | 0.103689 |
| C | lag(2 4) | 2-step | 5 | -0.092966 | 1.127787 | -0.295913 | 0.210852 | 0.153483 | 0.094691 |

Stata non-collapsed instrument structure:
```
Standard: D.(x z)          → 2 columns (Δx_t, Δz_t)
GMM-type: L(2/4).L.y       → 3 columns (y_{t-2}, y_{t-3}, y_{t-4})
                                          Total: 5
```

---

## 2. Stata Baseline: Collapsed (`collapse` as sub-option of `gmm()`)

From `abond_collapsed.log`:

| Run | Lag Spec | Step | Instruments | b_L1.y | b_x | b_z | se_L1.y | se_x | se_z |
|-----|----------|------|-------------|--------|-----|-----|---------|------|------|
| B | lag(2 4) | 1-step | **4** | -0.119842 | 1.125821 | -0.289741 | 0.246686 | 0.177270 | 0.104258 |
| C | lag(2 4) | 2-step | **4** | -0.119918 | 1.125117 | -0.289928 | 0.213669 | 0.151884 | 0.093735 |

Stata collapsed instrument structure:
```
Standard: D.(x z)                    → 2 columns
GMM-type: L(2/4).L.y collapsed       → 1 column (all 3 depths averaged)
                                                    Total: 4
```

Key observation: Stata's `collapse` averages across BOTH depths AND time periods, producing **1 GMM column** (not 3).

Note: `collapse` is invalid with `lag(2 2)` — Stata requires multiple GMM lags for collapsing to be meaningful.

---

## 3. oe Baseline: Collapsed (collapse=True, exogenous=["x","z"])

| Run | max_iv_lag | Step | Instruments | b_L1.y | b_x | b_z | se_L1.y | se_x | se_z |
|-----|-----------|------|-------------|--------|-----|-----|---------|------|------|
| A | 2 | 1-step | 3 | -0.118957 | 1.126002 | -0.289937 | 0.012220 | 0.007463 | 0.005590 |
| B | 4 | 1-step | **5** | -0.126472 | 1.120930 | -0.287164 | **0.010166** | **0.006980** | **0.006165** |
| C | 4 | 2-step | **5** | -0.123833 | 1.123870 | -0.288810 | **0.012205** | **0.007081** | **0.006165** |

oe collapsed instrument structure:
```
GMM-type: L(2/4).L.y       → 3 columns (one per depth: y_{t-2}, y_{t-3}, y_{t-4})
Standard: D.(x z)          → 2 columns
                                          Total: 5
```

---

## 4. Apples-to-Apples Comparison: Stata Collapsed vs oe Collapsed

### Run B (1-step, lag/max_iv_lag=4)

| Metric | Stata Collapsed | oe Collapsed | Ratio (oe/Stata) |
|--------|-----------------|--------------|-------------------|
| Instruments | **4** | **5** | — |
| b_L1.y | -0.119842 | -0.126472 | 1.055 |
| b_x | 1.125821 | 1.120930 | 0.996 |
| b_z | -0.289741 | -0.287164 | 0.991 |
| se_L1.y | 0.246686 | **0.010166** | **0.041** |
| se_x | 0.177270 | **0.006980** | **0.039** |
| se_z | 0.104258 | **0.006165** | **0.059** |

### Run C (2-step, lag/max_iv_lag=4)

| Metric | Stata Collapsed | oe Collapsed | Ratio (oe/Stata) |
|--------|-----------------|--------------|-------------------|
| Instruments | **4** | **5** | — |
| b_L1.y | -0.119918 | -0.123833 | 1.033 |
| b_x | 1.125117 | 1.123870 | 0.999 |
| b_z | -0.289928 | -0.288810 | 0.996 |
| se_L1.y | 0.213669 | **0.012205** | **0.057** |
| se_x | 0.151884 | **0.007081** | **0.047** |
| se_z | 0.093735 | **0.006165** | **0.066** |

---

## 5. Analysis

### 5a. Coefficient Match: IMPROVED but not exact

After fixing the exogenous instrument structure, coefficients for x and z match within 0.4-1%. The L1.y coefficient is within 3-6%. The remaining divergence is explained by:

**Instrument count discrepancy**: Stata collapsed has 4 instruments (1 GMM + 2 standard + 1 unknown), oe collapsed has 5 (3 GMM + 2 standard). Stata's `collapse` appears to average across depths into a single GMM column, while oe keeps one column per depth. This structural difference in the Z matrix explains the 5% coefficient divergence.

### 5b. SE Magnitude: NOT resolved — 24x too small

This is the critical finding. The SE discrepancy is **independent of collapsed-vs-collapsed**. Deep diagnostic of the sandwich VCV:

```
Z shape: (90, 5)     — 90 equations, 5 instruments
X shape: (90, 3)     — 90 equations, 3 regressors (L1.y, x, z)
Y shape: (90,)       — 90 equations

G = Z'X W X'Z:       (3, 3)    — correct
G_inv:               (3, 3)    — correct

Per-entity g_i:      (3,) each — correct (one g_i per entity, shape p)
S_g = Sum_i g_i g_i': (3, 3)   — correct (sum over 30 entities)

V_sandwich = G^-1 S_g G^-1: (3, 3)
  diag(V_sandwich) = [1.03e-04, 4.87e-05, 3.80e-05]
  se from V_sandwich = [0.01017, 0.00698, 0.00616]

Stata VCV diag = [0.06013, 0.03126, 0.01075]
Stata se = [0.24521, 0.17681, 0.10369]
```

The oe VCV is ~24x smaller than Stata's. The sandwich formula dimensions are correct:
- `g_i` shape = (p,) = (3,) — one per entity ✓
- `S_g` shape = (p, p) = (3, 3) — sum over 30 entities ✓
- `G` shape = (p, p) = (3, 3) ✓
- `V` shape = (p, p) = (3, 3) ✓

**The SE discrepancy is NOT caused by summing over N_obs instead of N_entities.** The loop correctly iterates over 30 entities, each contributing one g_i vector.

**Likely cause**: The weighting matrix `W = pinv(Z'Z)` is fundamentally different from Stata's. In 1-step GMM, Stata uses `W = (Z'Z)^{-1}` (the optimal GMM weighting matrix for homoskedastic errors). The oe code also uses `W = pinv(Z'Z)`, but the Z matrix structure differs (5 columns vs 4 columns in Stata collapsed), which changes `Z'Z` and therefore `W`.

However, even accounting for the instrument count difference, the SE ratio (24x) is far too large to be explained by a single extra column in Z. The issue is likely deeper in the GMM estimation — possibly the `small` option's effect on VCV, or a fundamental difference in how Stata computes the 1-step sandwich.

---

## 6. Remaining Questions

1. **What is Stata's 4th instrument in collapsed mode?** Stata shows 3 instruments (2 standard + 1 collapsed GMM) but reports 4. The Sargan dof=1 confirms 4 instruments with 3 regressors. What is the 4th column?

2. **Why is the SE 24x too small?** The sandwich dimensions are correct. The issue must be in the magnitudes of G, W, S_g, or their products. A scaling factor of ~24 (≈ sqrt(90/30) × something) suggests a possible N_obs vs N_entities confusion somewhere else, or a Stata `small` option effect on the VCV.

3. **Does the `small` option affect the VCV itself?** Stata's `small` uses t/F distributions for inference, but does it also modify the VCV computation? The oe implementation does not have a `small` option.

---

## 7. Files Modified

| File | Change |
|------|--------|
| `open_econs/models/linear/abond.py` | Removed constant, added exogenous partitioning, fixed L.y GMM instruments, added non-collapsed code path (identical to collapsed — needs fix) |
| `tests/stata/do/abond_collapsed.do` | New: Stata collapsed diagnostic |
| `tests/stata/do/abond_diag.py` | Updated: collapsed + non-collapsed comparison |
| `tests/stata/do/abond_deep_diag.py` | New: Z matrix structure, per-entity g_i, sandwich dimensions |
