from typing import Any

import numpy as np
import pandas as pd
from formulaic import Formula
from scipy.stats import chi2 as _chi2
from scipy.stats import norm as _norm

from open_econs.core.call_capture import capture_call as _capture_call


def _build_noncollapsed_gmm_block(
    var: np.ndarray, depth: int, T: int, lag_offset: int
) -> np.ndarray:
    """T × (T - depth - lag_offset) staircase block, zero-inactive columns omitted.

    Column k (0 … n_active-1) has a single non-zero at row
    ``j = k + depth + lag_offset`` with value ``var[k]``.
    All structurally zero columns (where j would be ≥ T) are dropped.

    Parameters
    ----------
    var : ndarray, shape (T,)
    depth : int
    T : int
    lag_offset : int
        ``lags`` for L.y, ``0`` for predetermined regressors.

    Returns
    -------
    block : ndarray, shape (T, max(0, T - depth - lag_offset))
    """
    n_active = T - depth - lag_offset
    if n_active <= 0:
        return np.zeros((T, 0))
    block = np.zeros((T, n_active))
    for k in range(n_active):
        j = k + depth + lag_offset
        block[j, k] = var[k]
    return block


def _tridiag_h_inv_block(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (diag, off_diag) of H^{-1} for an n×n tridiagonal H.

    H has 2 on the diagonal and -1 on the sub/super-diagonals.
    Its inverse has the closed form (0-indexed):
        H^{-1}_{ij} = (min(i,j)+1) * (n-max(i,j)) / (n+1)
    """
    diag = np.empty(n)
    off = np.empty(max(n - 1, 0))
    np1 = n + 1
    for i in range(n):
        diag[i] = (i + 1) * (n - i) / np1
    for i in range(n - 1):
        off[i] = (i + 1) * (n - i - 1) / np1
    return diag, off


def _build_h_inv(entity_counts: dict, n_eq: int, eq_entity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the block-diagonal H^{-1} matrix for difference GMM weighting.

    For first-differencing with iid errors, Var[Δe_i] = σ² H_i where each
    H_i is a tridiagonal (n_i × n_i) matrix with 2 on the diagonal and
    −1 on the sub/super-diagonals.  This reflects the MA(1) structure
    that differencing induces.

    Returns (diag, off_diag) of the tridiagonal H^{-1}.
    """
    diag = np.zeros(n_eq)
    off = np.zeros(max(n_eq - 1, 0))
    pos = 0
    for ent, n_i in entity_counts.items():
        if n_i < 1:
            continue
        d, o = _tridiag_h_inv_block(n_i)
        diag[pos: pos + n_i] = d
        off[pos: pos + len(o)] = o
        pos += n_i
    return diag, off


def _build_h(entity_counts: dict, n_eq: int, eq_entity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the block-diagonal H matrix (NOT its inverse) for difference GMM.

    H = M'M where M is the first-difference operator.  For usable equations
    (t >= min_j >= 2) the diagonal is 2 and the sub/super-diagonal is -1.
    (The very first time period t=0 has H diagonal 1, but it is never a usable
    equation so it never enters the moment conditions.)

    Returns (diag, off_diag) of the tridiagonal H.
    """
    diag = np.full(n_eq, 2.0)
    off = np.full(max(n_eq - 1, 0), -1.0)
    # Zero out off-diagonals at entity boundaries
    ent_arr = np.asarray(eq_entity)
    for k in range(n_eq - 1):
        if ent_arr[k] != ent_arr[k + 1]:
            off[k] = 0.0
    return diag, off


def _estimate_gmm(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    eq_entity: np.ndarray,
    step: str,
    robust: bool = False,
) -> dict[str, Any]:
    """Arellano-Bond difference GMM, mirroring xtabond2's Mata source (v3.7.2).

    Implements one-step / two-step / robust / Windmeijer-corrected VCEs exactly
    as the Mata code assembles them.  Key conventions:

    * One-step weighting ``A1 = (Z'HZ)^{-1}`` with ``H = M'M`` (first-difference
      operator, tridiagonal diag 2 / off -1).  Mata rescales ``A1, V1`` by
      ``sig2`` (Mata 418-419).
    * Two-step weighting ``A2 = S^{-1}`` with ``S = Σᵢ (Zᵢ'e1ᵢ)(Zᵢ'e1ᵢ)'`` built
      per-entity from the ONE-step residuals ``e1`` (Mata 450-460).
    * Same ``S`` is reused for ``A2`` (two-step weighting) and for the one-step
      robust sandwich ``V1robust`` (Mata 462-464).
    * Windmeijer two-step-robust VCE uses ``V2`` in two terms and the one-step
      robust ``V1robust`` in the ``D*V1robust*D'`` term (Mata 522-523).
    * Small-sample multiplier (Mata 562-565): one-step non-robust uses
      ``wttot/(wttot-k)``; all other cases use
      ``(NObs-1)/(NObs-k)·NGroups/(NGroups-1)``.
    * ``m2VZXA = -2·V·(ZX'·A)`` is built from the *pre*-small-sample V/A
      (Mata 549), while the V/sig2 fed into the AR test are the *post*-
      correction values (Mata 614) — operand timing must be preserved.
    """
    n_eq = Y.shape[0]
    N = float(len(np.unique(eq_entity)))
    p = X.shape[1]
    L = Z.shape[1]
    from collections import Counter
    entity_counts = dict(Counter(eq_entity.tolist()))

    ZtX = Z.T @ X  # (L, p)  == ZX'
    ZtY = Z.T @ Y  # (L,)

    # --- One-step weighting A1 = (Z'HZ)^{-1} ---
    H_diag, H_off = _build_h(entity_counts, n_eq, eq_entity)
    ZtHZ = 2.0 * (Z.T @ Z)
    ZH_off = Z[:-1] * H_off[:, None]
    ZtHZ += ZH_off.T @ Z[1:]
    ZtHZ += Z[1:].T @ ZH_off
    A1_raw = np.linalg.inv(ZtHZ)
    G1 = ZtX.T @ A1_raw @ ZtX
    V1_raw = np.linalg.inv(G1)
    b1 = V1_raw @ (ZtX.T @ A1_raw @ ZtY)
    e1 = Y - X @ b1

    wttot = float(n_eq)        # = NObs (no weights, difference GMM)
    NGroups = float(N)
    k = float(p)
    sig2 = float(e1 @ e1) / 2.0 / wttot
    # Mata 418-419: rescale A1, V1 by sig2 (one-step non-robust branch uses this)
    A1 = A1_raw / sig2
    V1 = V1_raw * sig2

    onestepnonrobust = (step == "one-step") and (not robust)

    # --- Per-entity S from ONE-step residuals (A2, V1robust, Hansen) ---
    S = np.zeros((L, L))
    for ent in np.unique(eq_entity):
        mask = eq_entity == ent
        ze = Z[mask].T @ e1[mask]
        S += np.outer(ze, ze)

    if onestepnonrobust:
        b = b1
        pV_pre = V1
        pA_pre = A1
        pe = e1
        pV = V1
    else:
        if robust:
            VXZA1 = V1 @ ZtX.T @ A1             # V1 * (ZX' A1)
            V1robust = VXZA1 @ S @ VXZA1.T
        A2 = np.linalg.inv(S)
        G2 = ZtX.T @ A2 @ ZtX
        V2 = np.linalg.inv(G2)
        b2 = V2 @ (ZtX.T @ A2 @ ZtY)
        e2 = Y - X @ b2
        if step == "two-step":
            sig2 = float(e2 @ e2) / 2.0 / wttot   # Mata 480: two-step sig2 from e2
        A2Ze = A2 @ (Z.T @ e2)
        if step == "one-step":
            b = b1
            pV_pre = V1
            pA_pre = A1
            pe = e1
            pV = V1robust if robust else V1
        else:  # two-step
            b = b2
            pV_pre = V2
            pA_pre = A2
            pe = e2
            if robust:
                # Windmeijer (2005) correction — Mata 510-523
                VXZA2 = V2 @ ZtX.T @ A2
                D = np.zeros((L, p))
                for ent in np.unique(eq_entity):
                    mask = eq_entity == ent
                    ze = Z[mask].T @ e1[mask]           # (L,)  Z_i' e1_i (one-step)
                    ZXi = Z[mask].T @ X[mask]           # (L, p)
                    # Mata 518: term1 = scalar (Z_i'e1_i · A2Ze) * ZXi ;
                    # term2 = outer(Z_i'e1_i, A2Ze'·ZXi) — per-row-varying scale.
                    s1 = ze @ A2Ze                      # scalar
                    term1 = s1 * ZXi                    # (L, p)
                    term2 = np.outer(ze, A2Ze @ ZXi)    # (L, p)
                    D += term1 + term2
                D = VXZA2 @ D                           # -> (p, p)
                V2robust = V2 + D @ V1robust @ D.T + 2.0 * D @ V2
                pV = V2robust
            else:
                pV = V2

    # --- Small-sample correction (Mata 562-565) ---
    # NB: Mata multiplies V by the *branch-specific* multiplier but multiplies
    # sig2 by `tmp = wttot/(wttot-k)` ALWAYS (line 565 uses `tmp`, not the
    # branch multiplier).  Keep the two multipliers separate.
    if onestepnonrobust:
        small_mult = wttot / (wttot - k)
    else:
        small_mult = ((wttot - 1.0) / (wttot - k)) * (NGroups / (NGroups - 1.0))
    V = pV * small_mult
    sig2 = sig2 * (wttot / (wttot - k))

    se = np.sqrt(np.maximum(np.diag(V), 0.0))

    # --- Hansen J (A1 for one-step, A2 for two-step) ---
    g_all = Z.T @ pe
    A_used = A1 if onestepnonrobust else A2
    dof_j = L - p
    J = float(g_all @ A_used @ g_all)
    p_j = float(1.0 - _chi2.cdf(J, dof_j)) if dof_j > 0 else float("nan")

    # --- Windmeijer/AR prep: m2VZXA from PRE-small V/A (Mata 549) ---
    m2VZXA = -2.0 * pV_pre @ (ZtX.T @ pA_pre)

    # pV_ar is the pre-small V passed to _ARTests (Stata line 614).
    # The reporting V and sig2 get the small multiplier; m2VZXA and pV_ar
    # use the base (uncorrected) values.
    return {
        "b": b, "se": se, "e": pe, "Z": Z, "X": X, "Y": Y,
        "sig2": sig2, "V": V,
        "J": J, "dof_j": dof_j, "p_j": p_j,
        "m2VZXA": m2VZXA, "pV": V, "pV_ar": pV,
        "eq_entity": eq_entity, "n_eq": n_eq, "N": N, "p": p, "L": L,
        "step": step, "robust": robust,
    }


def _build_H_ar(T: int, h: int = 3) -> np.ndarray:
    """T×T H = M'M for the first-difference transform (Mata ``_H(h)``).

    ``h=3`` (difference GMM default): ``M = I(T) - shift(I(T), 1)`` with
    ``M[0,0]=0``, so ``H[0,0]=0`` and ``H`` is tridiagonal with diagonal 2
    (except the last usable position) and off-diagonal -1.
    """
    M = np.eye(T) - np.eye(T, k=1)
    M[0, 0] = 0.0
    return M.T @ M


def _ar_test(
    e_by_entity: dict,
    Z_by_entity: dict,
    X_by_entity: dict,
    T_by_entity: dict,
    step: str,
    robust: bool,
    m2VZXA: np.ndarray,
    pV: np.ndarray,
    sig2: float,
    h: int = 3,
    n_lags: int = 2,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Arellano-Bond AR(1)/AR(2) tests, mirroring Mata ``_ARTests`` (1098-1167).

    Replicates the per-entity sums exactly:

    * two-step / robust (``onestepnonrobust == 0``):
        sum_wwli[i] = Σₜ rᵢₜ·rᵢ,ₜ₋ₗ ;  wHw = Σᵢ sum_wwli[i]²
        ZHw = Σᵢ (Zᵢ'eᵢ)·sum_wwli[i] ;  tmp = Σᵢ Xᵢ'·lag(rᵢ)
        ARz = Σᵢ sum_wwli[i] / √(wHw + tmp'·(m2VZXA·ZHw + V·tmp))
    * one-step non-robust: same numerator, but wHw and ZHw use the H-weighted
        lagged residual (Mata 1144-1154).

    Operand timing (Mata 549 vs 614): ``m2VZXA`` is built from the
    *pre*-small-sample V/A; ``pV`` here is the *base* (pre-small) V;
    ``sig2`` is the *post*-correction value (Stata multiplies H by the
    post-small ``sig2`` in the one-step non-robust branch).

    All input vectors (``e_by_entity``, ``Z_by_entity``, ``X_by_entity``) are
    expected to be *full T-length* per entity (T rows, first position
    hard-zeroed) — consistent with Stata's ``*pe`` and ``_ARTests`` conventions.
    """
    onestepnonrobust = (step == "one-step") and (not robust)
    L = next(iter(Z_by_entity.values())).shape[1]
    p = next(iter(X_by_entity.values())).shape[1]

    out = []
    for lag in range(1, n_lags + 1):
        sum_wwli_total = 0.0
        wHw = 0.0
        ZHw = np.zeros(L)
        tmp = np.zeros(p)
        for ent in e_by_entity:
            T_i = T_by_entity[ent]
            e_ent = e_by_entity[ent]           # full T-length, position 0 hard-zeroed
            wli = np.zeros(T_i)
            wli[lag:] = e_ent[:T_i - lag]       # lag (shift down, zero-fill at top)
            sum_wwli = float(e_ent @ wli)       # Σₜ rₜ · rₜ₋ₗ
            if onestepnonrobust:
                H = _build_H_ar(T_i, h)
                wHw += float(wli @ H @ wli) * sig2
                psiw = H @ wli * sig2
                ZHw += Z_by_entity[ent].T @ psiw
            else:
                wHw += sum_wwli ** 2
                ZHw += Z_by_entity[ent].T @ e_ent * sum_wwli
            tmp += X_by_entity[ent].T @ wli
            sum_wwli_total += sum_wwli
        denom = np.sqrt(wHw + tmp @ (m2VZXA @ ZHw + pV @ tmp))
        stat = sum_wwli_total / denom if denom > 0 else float("nan")
        pval = float(2.0 * (1.0 - _norm.cdf(abs(stat))))
        out.append((float(stat), pval))
    return tuple(out[0]), tuple(out[1])


def abond(
    formula: str,
    data: pd.DataFrame,
    entity: str,
    time: str,
    lags: int = 1,
    max_iv_lag: int | None = None,
    step: str = "two-step",
    exogenous: list[str] | None = None,
    collapse: bool = True,
    robust: bool = False,
) -> Any:
    """Arellano-Bond (1991) dynamic panel-data estimator (difference GMM).

    Estimates a dynamic panel model of the form

        y_it = a_1 y_{i,t-1} + ... + a_{lags} y_{i,t-lags}
               + b' x_it + mu_i + eps_it

    by first-differencing to remove the fixed effect ``mu_i`` and using deeper
    lags of the dependent variable (and of predetermined regressors) as GMM
    instruments.  Both the one-step and the efficient two-step estimators are
    available; the two-step standard errors use the Windmeijer (2005)
    small-sample correction.  Reports the Hansen J test of overidentifying
    restrictions and the Arellano-Bond AR(1)/AR(2) serial-correlation tests.

    Parameters
    ----------
    formula : str
        Two-sided formula, e.g. ``"y ~ x1 + x2"``.  The lagged dependent
        variable(s) are added automatically (``lags`` of them); the right-hand
        side lists the additional (predetermined / exogenous) regressors.
    data : pd.DataFrame
        Panel data.
    entity, time : str
        Column names for the panel entity and time indices.
    lags : int, default 1
        Number of own lags of the dependent variable to include.
    max_iv_lag : int, optional
        Maximum lag depth used as instruments (defaults to the longest
        available lag in the data).  Lower values reduce instrument
        proliferation for long panels.
    step : {"one-step", "two-step"}, default "two-step"
        GMM step.
    exogenous : list of str, optional
        Regressors that are *strictly* exogenous (analogous to Stata's
        ``iv()``).  These are instrumented with their own current-period
        differenced values rather than deeper lags.  If ``None``, all
        regressors are treated as predetermined (instrumented with deeper
        lags).
    collapse : bool, default True
        If True, use collapsed instruments (Roodman 2009): one instrument
        per lag depth rather than per lag depth x time period.  This reduces
        instrument proliferation and mitigates the "too many instruments"
        problem that biases two-step SEs downward in long panels.
    robust : bool, default False
        If True, use cluster-robust sandwich standard errors (analogous to
        Stata's ``robust`` option).  If False, use classical GMM standard
        errors based on σ̂² · (X'Z W Z'X)⁻¹ (the default in Stata's
        xtabond2 when ``robust`` is not specified).

    Returns
    -------
    ArellanoBondResult
    """
    if step not in ("one-step", "two-step"):
        raise ValueError("step must be 'one-step' or 'two-step'.")
    if lags < 1:
        raise ValueError("lags must be >= 1.")

    call = _capture_call(
        formula=formula, entity=entity, time=time, lags=lags,
        max_iv_lag=max_iv_lag, step=step, exogenous=exogenous,
        collapse=collapse, robust=robust,
    )

    formula_obj = Formula(formula)
    mm = formula_obj.get_model_matrix(data, na_action="drop")
    y_name = mm.lhs.columns[0]
    x_cols = [c for c in mm.rhs.columns if c != "Intercept"]

    df = data.loc[mm.rhs.index].copy()
    df["__y"] = mm.lhs[y_name].values.ravel()
    for c in x_cols:
        df["__x__" + c] = mm.rhs[c].values
    ent_vals = df[entity].values
    time_vals = df[time].values

    order = np.lexsort((time_vals, ent_vals))
    ent_sorted = ent_vals[order]
    y_sorted = df["__y"].values[order]
    x_sorted = {c: df["__x__" + c].values[order] for c in x_cols}

    entities: list[Any] = []
    y_by_e: dict[Any, np.ndarray] = {}
    x_by_e: dict[Any, dict[str, np.ndarray]] = {}
    for e_val in pd.unique(ent_sorted):
        mask = ent_sorted == e_val
        entities.append(e_val)
        y_by_e[e_val] = y_sorted[mask]
        x_by_e[e_val] = {c: x_sorted[c][mask] for c in x_cols}

    # Partition regressors into GMM-endogenous and strictly exogenous.
    # The lagged dependent variable(s) are always GMM-endogenous.
    exo_set = set(exogenous) if exogenous else set()
    gmm_cols = [c for c in x_cols if c not in exo_set]
    iv_cols = [c for c in x_cols if c in exo_set]

    min_j = max(lags + 1, 2)
    max_T = max(len(y_by_e[e]) for e in entities)
    if max_iv_lag is None:
        maxL = max_T - 1
    else:
        maxL = min(max_iv_lag, max_T - 1)
    depths = list(range(2, maxL + 1))
    if not depths:
        raise ValueError(
            "Not enough time periods to form Arellano-Bond instruments. "
            "Each entity needs at least 3 time periods."
        )

    # In collapsed mode, drop degenerate depths: a depth d requires t-d >= 0
    # for the instrument y_{t-d} to exist.  With usable equations at
    # t = min_j .. T-1, the number of valid time periods for depth d is
    # T - max(min_j, d).  If that count is < 2, Stata xtabond2 silently
    # drops the column (too few non-zero rows).  Only applies to collapsed.
    if collapse:
        valid_depths = [
            d for d in depths
            if max_T - max(min_j, d) >= 2
        ]
        if not valid_depths:
            raise ValueError(
                "No usable instrument depths after filtering degenerate columns."
            )
        depths = valid_depths

    # Instrument count:
    #   collapsed:   len(depths) * (1 + len(gmm_cols)) + len(iv_cols)
    #   uncollapsed: sum over entities of usable equations, expanded
    n_endog = 1 + len(gmm_cols)  # L.y + predetermined regressors

    Y_list: list[float] = []
    X_list: list[list[float]] = []
    Z_list: list[np.ndarray] = []
    eq_entity_list: list[Any] = []

    if collapse:
        # Collapsed instruments (Roodman 2009): for each lag depth, one
        # instrument column that averages across all available time periods
        # within each entity.  Reduces instrument count from
        # O(depths x T) to O(depths).
        n_instr = len(depths) * n_endog + len(iv_cols)
        for e_val in entities:
            y = y_by_e[e_val]
            xs = x_by_e[e_val]
            T = len(y)
            for j in range(min_j, T):
                dep = y[j] - y[j - 1]
                dyn_regs = [y[j - lag] - y[j - lag - 1]
                            for lag in range(1, lags + 1)]
                x_regs = [xs[c][j] - xs[c][j - 1] for c in x_cols]
                X_list.append(dyn_regs + x_regs)

                zrow = np.zeros(n_instr)
                col = 0
                # GMM instruments for L.y: Stata's `gmm(L.y, lag(a b))` uses
                # lags a..b of the variable L.y (= y_{t-lags}).  So the
                # instrument at depth `lag` is y_{t-lags-lag}, i.e. one lag
                # deeper than y_{t-lag}.  The initial observation y_0 is never
                # used (Stata requires t-lags-lag >= 1 implicitly via the lag
                # of L.y).
                for lag in depths:
                    idx = j - lags - lag
                    if idx >= 0:
                        zrow[col] = y[idx]
                    col += 1
                # GMM instruments for predetermined regressors
                for gmm_c in gmm_cols:
                    for lag in depths:
                        if j - lag >= 0:
                            zrow[col] = xs[gmm_c][j - lag]
                        col += 1
                # Standard instruments for exogenous regressors (current Δ)
                for iv_c in iv_cols:
                    zrow[col] = xs[iv_c][j] - xs[iv_c][j - 1]
                    col += 1

                Z_list.append(zrow)
                Y_list.append(dep)
                eq_entity_list.append(e_val)
    else:
        # Uncollapsed (full) instruments: one column per (depth × usable
        # time period) for each GMM base variable, matching Stata's
        # _MakeGMMinsts / _Explode block-diagonal construction.
        #   n_gmm_cols = n_endog × Σ depth (T_i - depth)  (varies per entity)
        # For simplicity, precompute per-entity GMM column count, then append
        # per-entity Z blocks.  IV columns remain non-expanding.
        for e_val in entities:
            y = y_by_e[e_val]
            xs = x_by_e[e_val]
            T_i = len(y)
            n_gmm_i = n_endog * sum(T_i - d for d in depths if T_i > d)
            n_iv_i = len(iv_cols)
            n_instr_i = n_gmm_i + n_iv_i

            Z_i = np.zeros((T_i, n_instr_i))
            col = 0
            for d in depths:
                # L.y block — lag_offset = lags
                blk = _build_noncollapsed_gmm_block(y, d, T_i, lag_offset=lags)
                nc = blk.shape[1]
                if nc:
                    Z_i[:, col:col + nc] = blk
                    col += nc
                # Predetermined-regressor blocks — lag_offset = 0
                for gmm_c in gmm_cols:
                    blk = _build_noncollapsed_gmm_block(
                        xs[gmm_c], d, T_i, lag_offset=0,
                    )
                    nc = blk.shape[1]
                    if nc:
                        Z_i[:, col:col + nc] = blk
                        col += nc
            # IV columns (exogenous, current Δ)
            for iv_c in iv_cols:
                for j in range(1, T_i):
                    Z_i[j, col] = xs[iv_c][j] - xs[iv_c][j - 1]
                col += 1

            # Extract usable equations (j ≥ min_j)
            for j in range(min_j, T_i):
                dep = y[j] - y[j - 1]
                dyn_regs = [y[j - lag] - y[j - lag - 1]
                            for lag in range(1, lags + 1)]
                x_regs = [xs[c][j] - xs[c][j - 1] for c in x_cols]
                X_list.append(dyn_regs + x_regs)
                Z_list.append(Z_i[j, :])
                Y_list.append(dep)
                eq_entity_list.append(e_val)

    if len(Y_list) == 0:
        raise ValueError(
            "No usable equations: every entity is too short for Arellano-Bond. "
            "Need at least 3 time periods per entity."
        )

    Y = np.array(Y_list, dtype=float)
    X = np.array(X_list, dtype=float)
    Z = np.array(Z_list, dtype=float)
    eq_entity = np.array(eq_entity_list)

    est = _estimate_gmm(Y, X, Z, eq_entity, step, robust=robust)

    coef_names = [f"L{lag}.{y_name}" for lag in range(1, lags + 1)] + x_cols
    coefficients = pd.Series(est["b"], index=coef_names)
    std_errors = pd.Series(est["se"], index=coef_names)
    z_stats = pd.Series(
        np.where(est["se"] > 0, est["b"] / est["se"], np.nan), index=coef_names,
    )
    p_values = pd.Series(
        2.0 * (1.0 - _norm.cdf(np.abs(z_stats.values))), index=coef_names,
    )
    conf_int = pd.DataFrame(
        {
            "lower": est["b"] - 1.96 * est["se"],
            "upper": est["b"] + 1.96 * est["se"],
        },
        index=coef_names,
    )

    # Per-entity FULL T-length vectors for the AR tests.
    # Stata's _ARTests receives a T-length residual per entity (first position
    # hard-zeroed by _Difference / touse).  Our estimation uses only the
    # "usable" equations (j >= min_j); we reconstruct the full T-length
    # residual here to match.
    p_ar = int(est["p"])
    L_ar = Z.shape[1]
    e_by_entity: dict[Any, np.ndarray] = {}
    Z_by_entity: dict[Any, np.ndarray] = {}
    X_by_entity: dict[Any, np.ndarray] = {}
    T_by_entity: dict[Any, int] = {}
    for e_val in entities:
        y_e = y_by_e[e_val]
        xs = x_by_e[e_val]
        T_i = len(y_e)
        b = est["b"]  # (p_ar,) coefficient vector

        # Full T-length X (first-differenced) and Y (differenced dep var).
        X_i = np.zeros((T_i, p_ar))
        Y_i = np.zeros(T_i)
        for j in range(1, T_i):
            col = 0
            for lag in range(1, lags + 1):
                # Differenced L{lag}.y at period j.
                # For j-lag == 0 the pre-sample value is treated as 0 (Stata's
                # _Difference convention: no observation before period 1).
                if j - lag >= 1:
                    X_i[j, col] = y_e[j - lag] - y_e[j - lag - 1]
                elif j - lag == 0:
                    X_i[j, col] = y_e[0]
                else:
                    X_i[j, col] = 0.0
                col += 1
            for c in x_cols:
                X_i[j, col] = xs[c][j] - xs[c][j - 1]
                col += 1
            Y_i[j] = y_e[j] - y_e[j - 1]

        # Full T × L Z matrix — dispatch construction to match estimation layout.
        if collapse:
            # Collapsed: one column per depth (same as estimation path).
            Z_i = np.zeros((T_i, L_ar))
            for j in range(1, T_i):
                col = 0
                for lag in depths:
                    idx = j - lags - lag
                    if idx >= 0:
                        Z_i[j, col] = y_e[idx]
                    col += 1
                for gmm_c in gmm_cols:
                    for lag in depths:
                        if j - lag >= 0:
                            Z_i[j, col] = xs[gmm_c][j - lag]
                        col += 1
                for iv_c in iv_cols:
                    Z_i[j, col] = xs[iv_c][j] - xs[iv_c][j - 1]
                    col += 1
        else:
            # Non-collapsed: reuse the same block-diagonal staircase
            # construction that the estimator used.
            n_gmm_i = n_endog * sum(T_i - d for d in depths if T_i > d)
            n_iv_i = len(iv_cols)
            Z_i = np.zeros((T_i, n_gmm_i + n_iv_i))
            col = 0
            for d in depths:
                blk = _build_noncollapsed_gmm_block(y_e, d, T_i, lag_offset=lags)
                nc = blk.shape[1]
                if nc:
                    Z_i[:, col:col + nc] = blk
                    col += nc
                for gmm_c in gmm_cols:
                    blk = _build_noncollapsed_gmm_block(
                        xs[gmm_c], d, T_i, lag_offset=0,
                    )
                    nc = blk.shape[1]
                    if nc:
                        Z_i[:, col:col + nc] = blk
                        col += nc
            for iv_c in iv_cols:
                for j in range(1, T_i):
                    Z_i[j, col] = xs[iv_c][j] - xs[iv_c][j - 1]
                col += 1

        # Full residual: (Y - X·b) with position 0 hard-zeroed.
        e_full = Y_i - X_i @ b
        # Zero out periods t < min_j — these are not usable equations
        # (no GMM instruments available), matching Stata's _ARTests convention.
        e_full[:min_j] = 0.0
        X_i[:min_j] = 0.0
        Z_i[:min_j] = 0.0
        e_by_entity[e_val] = e_full
        Z_by_entity[e_val] = Z_i
        X_by_entity[e_val] = X_i
        T_by_entity[e_val] = T_i

    # Pass PRE-small V (pV_ar) and POST-small sig2 to _AR tests,
    # matching Stata's _ARTests operand timing (Mata 549 vs 614).
    ar1, ar2 = _ar_test(
        e_by_entity, Z_by_entity, X_by_entity, T_by_entity,
        step, robust, est["m2VZXA"], est["pV_ar"], est["sig2"],
    )

    from open_econs.core.panel_results import ArellanoBondResult

    return ArellanoBondResult(
        formula=formula,
        coefficients=coefficients,
        std_errors=std_errors,
        z_stats=z_stats,
        p_values=p_values,
        conf_int=conf_int,
        step=step,
        lags=lags,
        n_entities=int(len(entities)),
        n_obs=int(len(Y)),
        n_instruments=int(Z.shape[1]),
        hansen_j=est["J"],
        hansen_j_pvalue=est["p_j"],
        hansen_j_dof=int(est["dof_j"]),
        sig2=float(est["sig2"]),
        ar1_stat=ar1[0],
        ar1_pvalue=ar1[1],
        ar2_stat=ar2[0],
        ar2_pvalue=ar2[1],
        call=call,
    )
