from typing import Any

import numpy as np
import pandas as pd
from formulaic import Formula
from scipy.stats import chi2 as _chi2
from scipy.stats import norm as _norm

from open_econs.core.call_capture import capture_call as _capture_call


def _estimate_gmm(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    eq_entity: np.ndarray,
    step: str,
) -> dict[str, Any]:
    """Arellano-Bond difference GMM with Windmeijer (2005) two-step SEs.

    Y : (n_eq,) dependent (differenced) vector
    X : (n_eq, p) regressor (differenced) matrix
    Z : (n_eq, L) instrument matrix (0 where instrument unavailable)
    eq_entity : (n_eq,) entity label per equation (defines the moment clusters)
    """
    n_eq = Y.shape[0]
    N = float(len(np.unique(eq_entity)))
    p = X.shape[1]
    L = Z.shape[1]

    ZtX = Z.T @ X  # (L, p)
    ZtY = Z.T @ Y  # (L,)

    # GMM moment form (summed over entities, no spurious 1/N scaling):
    #   g_i = X_i' Z_i W Z_i' e_i   (p,)
    #   G   = Sum_i X_i' Z_i W Z_i' X_i
    #   Var(b) = G^{-1} (Sum_i g_i g_i') G^{-1}
    ZtZ = Z.T @ Z  # (L, L)
    if step == "one-step":
        W = np.linalg.pinv(ZtZ) if L > 0 else np.eye(0)
        b1 = None
        e1 = None
    else:
        # One-step residuals drive the efficient two-step weighting matrix.
        G1 = ZtX.T @ np.linalg.pinv(ZtZ) @ ZtX
        b1 = np.linalg.inv(G1) @ (ZtX.T @ np.linalg.pinv(ZtZ) @ ZtY)
        e1 = Y - X @ b1
        S1 = (Z * e1[:, None]).T @ (Z * e1[:, None])  # (L, L)
        W = np.linalg.pinv(S1)
    G = ZtX.T @ W @ ZtX
    g_sum = ZtX.T @ W @ ZtY
    b = np.linalg.inv(G) @ g_sum
    e = Y - X @ b

    # Entity-clustered middle sandwich Sum_i g_i g_i'.
    S_g = np.zeros((p, p))
    for ent in np.unique(eq_entity):
        mask = eq_entity == ent
        Zc = Z[mask]  # (n_eq_e, L)
        Xc = X[mask]  # (n_eq_e, p)
        ec = e[mask]  # (n_eq_e,)
        Zte = Zc.T @ ec  # (L,)
        XtZ = Xc.T @ Zc  # (p, L)
        gi = XtZ @ W @ Zte  # (p,)
        S_g += np.outer(gi, gi)

    G_inv = np.linalg.inv(G)
    if step == "two-step":
        # Windmeijer (2005) small-sample correction.  Written as the explicit
        # sum of rank-one PSD terms so the middle sandwich stays PSD even when
        # W2 is ill-conditioned (deep instrument sets), avoiding tiny negative
        # eigenvalues from the algebraically-equivalent (1/N) g_sum g_sum' form.
        g_bar = g_sum / N
        S_g = np.zeros((p, p))
        for ent in np.unique(eq_entity):
            mask = eq_entity == ent
            Zc = Z[mask]
            Xc = X[mask]
            ec = e[mask]
            Zte = Zc.T @ ec
            XtZ = Xc.T @ Zc
            gi = XtZ @ W @ Zte
            gd = gi - g_bar
            S_g += np.outer(gd, gd)
    V = G_inv @ S_g @ G_inv
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
        Regressors that are *strictly* exogenous.  Treated like predetermined
        regressors here (instrumented with their own deeper lags).

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
    for e in pd.unique(ent_sorted):
        mask = ent_sorted == e
        entities.append(e)
        y_by_e[e] = y_sorted[mask]
        x_by_e[e] = {c: x_sorted[c][mask] for c in x_cols}

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

    n_instr = len(depths) * (1 + len(x_cols))

    Y_list: list[float] = []
    X_list: list[list[float]] = []
    Z_list: list[np.ndarray] = []
    eq_entity_list: list[Any] = []

    for e in entities:
        y = y_by_e[e]
        xs = x_by_e[e]
        T = len(y)
        for j in range(min_j, T):
            dep = y[j] - y[j - 1]
            dyn_regs = [y[j - lag] - y[j - lag - 1] for lag in range(1, lags + 1)]
            x_regs = [xs[c][j] - xs[c][j - 1] for c in x_cols]
            X_list.append(dyn_regs + x_regs)

            zrow = np.zeros(n_instr)
            col = 0
            for lag in depths:
                if j - lag >= 0:
                    zrow[col] = y[j - lag]
                col += 1
                for c in x_cols:
                    if j - lag >= 0:
                        zrow[col] = xs[c][j - lag]
                    col += 1
            Z_list.append(zrow)
            Y_list.append(dep)
            eq_entity_list.append(e)

    if len(Y_list) == 0:
        raise ValueError(
            "No usable equations: every entity is too short for Arellano-Bond. "
            "Need at least 3 time periods per entity."
        )

    Y = np.array(Y_list, dtype=float)
    X = np.array(X_list, dtype=float)
    Z = np.array(Z_list, dtype=float)
    eq_entity = np.array(eq_entity_list)

    est = _estimate_gmm(Y, X, Z, eq_entity, step)

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
    for e in entities:
        T = len(y_by_e[e])
        n_eq_e = max(0, T - min_j)
        e_by_entity[e] = est["e"][pos: pos + n_eq_e]
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
