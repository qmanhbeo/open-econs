from typing import Any

import numpy as np
import pandas as pd
from formulaic import Formula
from scipy.stats import chi2 as _chi2
from scipy.stats import norm as _norm

from open_econs.core.call_capture import capture_call as _capture_call


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

    Per Roodman (2009) xtabond2 h(3) default: H = M'M + I where M is the
    first-difference operator.  For usable equations (t >= min_j >= 2),
    the diagonal is 3 (not 2), because M'M has 2 on diagonal and I adds 1.

    Returns (diag, off_diag) of the tridiagonal H.
    """
    diag = np.full(n_eq, 3.0)
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
    """Arellano-Bond difference GMM with Windmeijer (2005) two-step SEs.

    Y : (n_eq,) dependent (differenced) vector
    X : (n_eq, p) regressor (differenced) matrix
    Z : (n_eq, L) instrument matrix (0 where instrument unavailable)
    eq_entity : (n_eq,) entity label per equation (defines the moment clusters)
    robust : if True, use cluster-robust sandwich VCV (G⁻¹ S_g G⁻¹).
             if False, use classical one-step VCV (σ̂² · G⁻¹).
    """
    n_eq = Y.shape[0]
    N = float(len(np.unique(eq_entity)))
    p = X.shape[1]
    L = Z.shape[1]

    ZtX = Z.T @ X  # (L, p)
    ZtY = Z.T @ Y  # (L,)

    # Entity-level equation counts (needed for block-diagonal H).
    from collections import Counter
    entity_counts = dict(Counter(eq_entity.tolist()))

    # GMM moment form (summed over entities, no spurious 1/N scaling):
    #   g_i = X_i' Z_i W Z_i' e_i   (p,)
    #   G   = Sum_i X_i' Z_i W Z_i' X_i
    #   Var(b) = G^{-1} (Sum_i g_i g_i') G^{-1}

    # H-matrix for difference GMM (Roodman xtabond2 h(3) default):
    # Block-diagonal with tridiagonal (2, -1) blocks reflecting the
    # MA(1) serial correlation that first-differencing induces.
    # W = (Z'HZ)^{-1} is the one-step weighting matrix.
    H_diag, H_off = _build_h(entity_counts, n_eq, eq_entity)
    # Z' H Z for tridiagonal H:
    #   diagonal: 2 * Z'Z
    #   off-diag: -1 * (Z_r' Z_{r+1} + Z_{r+1}' Z_r)
    ZtHZ = 3.0 * (Z.T @ Z)  # (L, L) diagonal contribution (H has 3 on diag for usable eqs)
    # Super/sub-diagonal contribution (vectorised), zeroed at entity boundaries
    ZH_off = Z[:-1] * H_off[:, None]  # (n_eq-1, L) — H_off is -1 within entity, 0 at boundary
    ZtHZ += ZH_off.T @ Z[1:]  # (L, L)
    ZtHZ += Z[1:].T @ ZH_off  # (L, L)

    if step == "one-step":
        W = np.linalg.pinv(ZtHZ) if L > 0 else np.eye(0)
    else:
        # Two-step: first step uses h(3)-weighted one-step W.
        W1 = np.linalg.pinv(ZtHZ) if L > 0 else np.eye(0)
        G1 = ZtX.T @ W1 @ ZtX
        b1 = np.linalg.inv(G1) @ (ZtX.T @ W1 @ ZtY)
        e1 = Y - X @ b1
        S1 = (Z * e1[:, None]).T @ (Z * e1[:, None])  # (L, L)
        W = np.linalg.pinv(S1)

    G = ZtX.T @ W @ ZtX
    g_sum = ZtX.T @ W @ ZtY
    G_inv = np.linalg.inv(G)
    b = G_inv @ g_sum
    e = Y - X @ b

    if robust:
        # Cluster-robust sandwich VCV: V = G⁻¹ S_g G⁻¹
        S_g = np.zeros((p, p))
        for ent in np.unique(eq_entity):
            mask = eq_entity == ent
            Zc = Z[mask]
            Xc = X[mask]
            ec = e[mask]
            Zte = Zc.T @ ec
            XtZ = Xc.T @ Zc
            gi = XtZ @ W @ Zte
            S_g += np.outer(gi, gi)

        if step == "two-step":
            # Windmeijer (2005) small-sample correction for two-step GMM.
            D = np.zeros((p, p))
            for ent in np.unique(eq_entity):
                mask = eq_entity == ent
                Zc = Z[mask]
                Xc = X[mask]
                ec = e[mask]
                Zte = Zc.T @ ec
                XtZ = Xc.T @ Zc
                gi = XtZ @ W @ Zte
                D += XtZ @ W @ Zc.T @ Xc @ G_inv @ gi
            D = -G_inv @ D

            V_sandwich = G_inv @ S_g @ G_inv
            V = V_sandwich + (1.0 / N) * (
                D @ V_sandwich + V_sandwich @ D.T + D @ G_inv @ D.T
            )
        else:
            V = G_inv @ S_g @ G_inv
    else:
        # Classical (non-robust) one-step VCV: V = σ̂² · G⁻¹
        # Roodman (2009) / xtabond2: σ̂² = Σê² / (df · (2 - (h==1)))
        # For h=3 (default difference GMM): factor = 2
        # df = n_eq - p  (number of equations minus regressors)
        h_factor = 2.0  # h=3 default: MM' blocks have trace/n ≈ 2
        df = float(n_eq - p)
        sig2 = float(e @ e) / (h_factor * df)
        V = sig2 * G_inv

    se = np.sqrt(np.maximum(np.diag(V), 0.0))

    # Hansen J test of overidentifying restrictions.
    g_all = Z.T @ e  # (L,)
    S_h = (Z * e[:, None]).T @ (Z * e[:, None])  # (L, L)
    dof_j = L - p
    J = float(g_all @ np.linalg.inv(S_h) @ g_all)
    p_j = float(1.0 - _chi2.cdf(J, dof_j)) if dof_j > 0 else float("nan")

    return {
        "b": b, "se": se, "e": e, "Z": Z, "X": X, "Y": Y,
        "J": J, "dof_j": dof_j, "p_j": p_j,
        "eq_entity": eq_entity, "n_eq": n_eq, "N": N, "p": p, "L": L,
    }


def _ar_test(e_by_entity: dict, p_lag: int) -> tuple[float, float]:
    """Arellano-Bond serial-correlation test on first-differenced residuals.

    Under H0 of no AR(p) in the level residuals the statistic is asymptotically
    standard normal. Returns (statistic, two-sided p-value).
    """
    num = 0.0
    den = 0.0
    for resid in e_by_entity.values():
        r = resid
        if len(r) <= p_lag:
            continue
        for t in range(p_lag, len(r)):
            num += r[t] * r[t - p_lag]
            den += (r[t] * r[t - p_lag]) ** 2
    if den <= 0:
        return (float("nan"), float("nan"))
    stat = num / np.sqrt(den)
    pval = float(2.0 * (1.0 - _norm.cdf(abs(stat))))
    return (float(stat), pval)


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
                # GMM instruments for L.y: y_{t-lag}
                for lag in depths:
                    if j - lag >= 0:
                        zrow[col] = y[j - lag]
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
        # Uncollapsed (full) instruments: one column per (depth x time period)
        # for GMM instruments, plus one column per time period for standard
        # instruments.  This matches Stata xtabond2's default behavior.
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
                # GMM instruments for L.y: y_{t-lag}
                for lag in depths:
                    if j - lag >= 0:
                        zrow[col] = y[j - lag]
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

    # Per-entity residuals (in time order) for the AR tests.
    e_by_entity: dict[Any, np.ndarray] = {}
    pos = 0
    for e_val in entities:
        T = len(y_by_e[e_val])
        n_eq_e = max(0, T - min_j)
        e_by_entity[e_val] = est["e"][pos: pos + n_eq_e]
        pos += n_eq_e

    ar1 = _ar_test(e_by_entity, 1)
    ar2 = _ar_test(e_by_entity, 2)

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
        ar1_stat=ar1[0],
        ar1_pvalue=ar1[1],
        ar2_stat=ar2[0],
        ar2_pvalue=ar2[1],
        call=call,
    )
