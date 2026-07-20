# HAND-OFF: system-GMM (Blundell-Bond) one-step Z + coupled weight H

Investigator: numerical-investigation subagent. Status: **UNRESOLVED at 1e-6** — framework
nailed, exact instrument-column formulas NOT yet matched to A1.csv.

## 1. Context
Goal: reproduce Stata `xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff))
gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) twostep small` one-step weight
`A1 = (Z' H Z)^{-1}` (11x11, rank 10) and coefficient vector `b` to tolerance 1e-6.
Ground truth: `tests/stata/generate-fixtures/archive/A1.csv` (header row, 11x11),
`A2.csv` (two-step), and b-targets recorded in the task brief.

Data: `tests/stata/fixtures/inputs/df_panel.csv` — 150 obs, 30 entities, T=0..4 (5 periods),
balanced. Diff equation uses usable periods t in {2,3,4} (3 per entity) -> 90 stacked obs.

## 2. Key known facts (high confidence)
- A1.csv is 11x11. Its 10th row/col (index 9, labelled c10) is EXACTLY zero, so the
  underlying `Z' H Z` is 11x11 rank 10. ONE of the 11 instrument columns is identically
  zero (a degenerate/dropped instrument). This matches Stata's printed
  "Number of instruments = 10" with e(j0)=11 (the extra slot is the zero column).
- True weight `W_true = pinv(A1)` is symmetric, diag in [33,149]. Notable EXACT-ZERO
  fingerprints in W_true:
    * index 4 row couples ONLY to {0,1,4,10} (zeros at 2,3,5,6,7,8,9) -> a diff-eq instrument.
    * index 9 (global) is all-zero -> the degenerate column.
    * W_true[4,5] = 0 (Dx orthogonal to Dz in Stata), W_true[2,4]=W_true[3,4]=0.
- Coupled H confirmed form (per xtabond2.ado): per entity
    H = [[M'M, M']; [M, I]]   with M the first-difference operator.
  Residuals stacked as [e_diff (3); e_level (3)] per entity, block-diag across entities.
- Z is BLOCK-DIAGONAL across equations, NOT [Z;Z]: diff-equation instruments feed e_diff,
  level-equation instruments feed e_level. This is mandatory to reach 11 columns and the
  correct block coupling. So:
    W = Z_full' H Z_full,  Z_full = blockdiag(Zd, Zl) per entity,
    W11 = Zd' M'M Zd,  W12 = Zd' M' Zl,  W22 = Zl' Zl.
- Column ordering in A1 follows Stata: all DIFF-equation instruments first, then all
  LEVEL-equation instruments. With diff GMM (4) + Dx + Dz = 6 diff cols, and level GMM (2)
  + x + z + const = 5 level cols -> 11 total. (Diff GMM = 4 cols is the working hypothesis:
  "L.L.y collapsed" + "L(2/4).L.y collapsed" treated as 4 lag columns.)
- M = 3x3 first-difference (M[i,i]=1, M[i,i-1]=-1) operating within the 3 usable diff
  periods. M00 (first diagonal) tried as 1 and 2; neither matched. The 5->3 "full levels"
  M variant also tried; builder needs fixing for that path (size mismatch) before trusting it.

## 3. Actions taken
- Wrote `tests/stata/generate-fixtures/archive/resolve_z.py` (self-contained, no library
  edits). It: loads df_panel.csv; builds per-entity M (3x3 or full 5->3); builds block-
  diagonal Z_full from a (Zd, Zl) pair; computes W=Zf'H Zf and A1=pinv(W); reports
  max-relative deviation vs A1.csv across many Z-definition variants and M settings.
- Established definitively that Z must be block-diagonal (diff vs level) and that the
  diff block has 6 cols, level block 5 cols (total 11, rank 10).
- RULED OUT: Z=[Z;Z] (gives wrong count/structure); entity-constant "collapsed" columns
  (make W11 hugely inflated & wrong-sign); the nonstandard M (2/-1) (breaks symmetry).
- Closest variants (diff4a/diff4b, M00=1, 3x3) still differ from A1 by ~1e10 relative —
  i.e. the diff-GMM and/or diff-IV column FORMULAS are still wrong, not merely scaled.
  Evidence: true W11[4,5]=0 (Dx⊥Dz) but every candidate gives large W11[4,5]; true
  W11[0,1]≈2 while candidates give 100s (diff-GMM cols far too collinear).

## 4. What remains (for the implementing agent)
1. Nail the EXACT diff-equation instrument columns. Strong candidates to test next:
   - diff GMM: are the collapsed columns PERIOD-VARYING lag values (y_{t-s}) summed in the
     MOMENT (not as entity-constant columns in the weight)? Test lags {2,3,4} and {1,2,3}
     and {2,3,4,5}. Note the "L.L.y collapsed" vs "L(2/4).L.y collapsed" split — one of
     these may be the differentiated L.y (y_{t-1}-y_{t-2}), not a plain level lag.
   - diff IV (x,z): Stata's true columns are NOT matching my D.x/D.z (Gram 341 vs 40, and
     Dx should be orthogonal to Dz). Re-derive from xtabond2.ado whether eq(diff) iv(x z)
     uses LEVEL x,z (not differenced) as instruments for the difference equation, and
     whether they are within-entity de-meaned. THIS is the most likely culprit.
2. Resolve the normalization: W may be divided by N (30) or obs count; A1=pinv(W) scale
   must match. But scaling alone cannot explain the wrong off-diagonal structure, so this
   is secondary to (1).
3. Once A1 matches to 1e-6: estimate b one-step = (X'Z A1 Z'X)^{-1} X'Z A1 Z'y, and two-step
   with S from one-step residuals. Compare to targets (1s: b_Ly=0.110421, b_x=1.156291,
   b_z=-0.603776, b_cons=0.061418; 2s: 0.009464,1.134976,-0.442064,0.090758).
4. Implement in abond.py's system branch; add parity tests; per rules 3/15 expose the
   collapsed/ordering choices as toggles and cover both. Keep system=False byte-identical.

## 5. Concerns / footguns
- The collapsed-lag(2/4) interpretation is the core ambiguity. xtabond2 `collapse` may
  mean "one column per lag, summed across time in the moment" (period-varying column in the
  weight) OR "entity-constant". A1.csv is the ONLY arbiter — trust it, not the log prose.
- zrank 10 vs A1 dimension 11: do not pad blindly; the zero 10th column is a real dropped
  instrument (Stata keeps the slot). pinv handles it; keep the 11th column literally zero.
- One-step AR(1)/AR(2) must be non-NaN; hansen_j is NaN for the c_1s_nr (one-step
  non-robust) case in Stata — preserve that.
- non-collapsed GMM -> NotImplementedError (out of scope; only the collapsed case here).
- R parity deferred. system=False (diff-only) must remain byte-identical to existing tests.
- Subagent over-run was the prior failure mode; this investigation was bounded and stopped
  at the wall (framework solved, column formulas unsolved) rather than guessing further.
- resolve_z.py currently has a known bug in the full-M (5->3) path (H 8x8 vs Zf 6x6 size
  mismatch) — only fix if that path is pursued; the 3x3 path runs.
