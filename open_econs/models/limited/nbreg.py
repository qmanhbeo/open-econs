"""Negative binomial regression (``nbreg``) — NB1 (mean dispersion) and NB2
(constant dispersion), pooled or with high-dimensional fixed effects.

Backend
------
There is no ``pyfixest.fenegbin`` in pyfixest 0.60.0, and Stata's base ``nbreg``
has no fixed-effect absorption (Stata NB-FE lives only in ``xtnbreg, fe``).  We
therefore **hand-roll** the NB estimator inside OE's HDFE demeaning core
(alternating projections, the same IRLS engine that powers ``oe.poisson``),
rather than wrapping an external package.

Parity targets (verified, see ``methodology/limited/nbreg.md``)
---------------------------------------------------------------
* **NB2 (``dispersion="const"``, default):** the standard NB2 gamma-Poisson
  mixture, ``Var = mu + alpha*mu**2``.  Matches **R ``MASS::glm.nb``** and
  **statsmodels NB2** to ``1e-6`` on coefficients, ``alpha = 1/theta``, and
  log-likelihood.  It also matches **Stata ``nbreg, dispersion(mean)``** (Stata's
  "mean" dispersion happens to coincide with the textbook NB2 MLE on the parity
  dataset — see the methodology note on the Stata ``constant`` vs ``mean``
  divergence).
* **NB1 (``dispersion="mean"``):** ``Var = mu * (1 + alpha)`` (Hilbe NB1).
  Matches the Hilbe NB1 MLE.  R ``fixest::fenegbin`` is NB2-only, so NB1 FE has
  no R-FE reference; pooled NB1 is cross-checked internally.

``alpha`` is the overdispersion parameter (Stata ``e(alpha)``).  We additionally
report ``lnalpha = log(alpha)`` (Stata ``e(lnalpha)``) and ``theta = 1/alpha``
(R ``glm.nb`` / ``fenegbin`` ``theta`` / NB2 size parameter).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from open_econs.core.call_capture import capture_call as _capture_call
from open_econs._internal import errors
from open_econs.core.results import NegBinResult
from open_econs.core.cov_type import validate_cov_type


def nbreg(
    formula: str,
    data: pd.DataFrame,
    fixed_effects: list[str] | None = None,
    cluster: str | list[str] | None = None,
    cov_type: str = "HC1",
    dispersion: str = "const",
    vcov_backend: str = "fixest",
    offset: str | None = None,
    weights: str | None = None,
) -> NegBinResult:
    """Estimate a negative binomial regression (NB1 / NB2).

    Hand-rolled NB estimator (no external NB backend exists in pyfixest 0.60.0;
    Stata base ``nbreg`` has no FE absorption).  Fixed effects are absorbed via
    the alternating-projections IRLS core shared with :func:`open_econs.poisson`;
    the overdispersion ``alpha`` is profiled out by 1-D likelihood maximization.

    Parameters
    ----------
    formula : str
        Two-sided formula, e.g. ``"y ~ x1 + x2"``. Do **not** put the
        fixed-effect indicators in the formula; use *fixed_effects*.
    data : pd.DataFrame
        Data with all formula variables plus FE / cluster / offset columns.
    fixed_effects : list of str, optional
        Column names of N-way fixed effects to absorb. If omitted, a pooled NB
        (no FE) is estimated, matching Stata ``nbreg`` / R ``glm.nb``.
    cluster : str or list of str, optional
        Column(s) for cluster-robust (CRV1) standard errors; a list requests
        multi-way clustering. Takes precedence over *cov_type*.
    cov_type : str, default "HC1"
        Heteroskedasticity-robust estimator used when *cluster* is not set.
        One of ``"nonrobust"``, ``"HC0"``, ``"HC1"``, ``"HC2"``, ``"HC3"``.
    dispersion : {"const", "mean"}, default "const"
        Overdispersion structure (Stata ``nbreg, dispersion(...))`` naming).

        - ``"const"`` -> **NB2** (DEFAULT): ``Var = mu + alpha*mu**2``.
        - ``"mean"``  -> **NB1**: ``Var = mu * (1 + alpha)``.
    vcov_backend : {"fixest", "stata"}, default "fixest"
        Small-sample convention for the reported variance (rule-15 toggle).
        Only rescales the cluster/robust variance — point estimates, deviance,
        and log-likelihood are identical.

        - ``"fixest"`` (default): ``k_adj=True, G_adj=True``.
        - ``"stata"``: ``k_adj=False, G_adj=True, k_fixef="none"``
          (ppmlhdfe-style adjustment).
    offset : str, optional
        Column name of an exposure offset (log-exposure), coefficient fixed at 1.
        ``mu = exp(x'b + fe + offset)``.
    weights : str, optional
        Column name of frequency weights (applied as analytic weights).

    Returns
    -------
    NegBinResult
        Immutable result with ``.coefficients`` (log scale), ``.alpha()`` /
        ``.lnalpha()`` / ``.theta()`` overdispersion, ``.tidy()``,
        ``.summary()``, ``.margins()``, ``.predict()``, ``.vcov()``.

    Examples
    --------
    >>> import open_econs as oe
    >>> r = oe.nbreg("y ~ x1 + x2", data=df)                      # pooled NB2
    >>> r = oe.nbreg("y ~ x1 + x2", data=df, dispersion="mean")   # pooled NB1
    >>> r = oe.nbreg("y ~ x1 + x2", data=df, fixed_effects=["firm", "year"],
    ...              cluster="firm")                              # NB2 with FE
    """
    call = _capture_call(
        formula=formula, fixed_effects=fixed_effects, cluster=cluster,
        cov_type=cov_type, dispersion=dispersion, vcov_backend=vcov_backend,
        offset=offset, weights=weights,
    )

    if vcov_backend not in ("fixest", "stata"):
        raise ValueError(
            f"vcov_backend must be 'fixest' or 'stata', got {vcov_backend!r}."
        )
    cov_type = validate_cov_type(
        cov_type,
        accepted={"nonrobust", "HC0", "HC1", "HC2", "HC3"},
        estimator="nbreg()",
    )
    if dispersion not in ("const", "mean"):
        raise ValueError(
            f"dispersion must be 'const' (NB2) or 'mean' (NB1), got "
            f"{dispersion!r}."
        )

    fe_parts: list[str] = list(fixed_effects) if fixed_effects else []
    has_fe = len(fe_parts) > 0

    lhs = formula.split("~", 1)[0].strip()
    rhs_formula = formula.split("~", 1)[1].strip()
    x_terms = [t.strip() for t in rhs_formula.split("+") if t.strip() and t.strip() != "1"]

    needed = [lhs, *x_terms, *fe_parts]
    if isinstance(cluster, str):
        needed.append(cluster)
    elif isinstance(cluster, list):
        needed.extend(cluster)
    if offset is not None:
        needed.append(offset)
    if weights is not None:
        needed.append(weights)
    missing = [c for c in dict.fromkeys(needed) if c not in data.columns]
    if missing:
        raise errors.missing_column_error(missing[0], data.columns.tolist())

    used_cols = list(dict.fromkeys(needed))
    original_n = len(data)
    work = data[used_cols].dropna().reset_index(drop=True)
    dropped = original_n - len(work)
    if dropped > 0:
        import warnings as _w
        _w.warn(
            errors.rows_dropped_warning(dropped, original_n, []),
            RuntimeWarning,
            stacklevel=3,
        )
    if len(work) == 0:
        raise errors.empty_data_error(original_n, dropped, [])

    y = work[lhs].astype(float).values
    X = work[x_terms].astype(float).values
    if X.shape[1] == 0:
        raise ValueError("nbreg() requires at least one regressor in the RHS.")
    # Pooled NB includes an intercept; with FE absorbed the intercept is
    # collinear with the FE and is omitted (matches R fixest::fenegbin).
    if has_fe:
        design_names = list(x_terms)
    else:
        X = np.column_stack([np.ones(len(work)), X])
        design_names = ["Intercept", *x_terms]
    if offset is not None:
        off = work[offset].astype(float).values
    else:
        off = np.zeros(len(work))
    if weights is not None:
        wts = work[weights].astype(float).values
    else:
        wts = np.ones(len(work))

    fe_groups = [work[c].astype("category").cat.codes.values for c in fe_parts]
    has_fe = len(fe_parts) > 0

    res = _fit_nb(
        y=y, X=X, fe_groups=fe_groups, has_fe=has_fe, dispersion=dispersion,
        wts=wts, off=off,
    )
    beta = res["beta"]
    fe_effects = res["fe_effects"]
    alpha = res["alpha"]
    eta = X @ beta + off
    if has_fe:
        eta = eta + _add_fe(fe_effects, fe_groups)
    mu = np.exp(eta)

    n = len(y)
    k = len(x_terms)
    n_absorbed = _count_absorbed_dof(work, fe_parts) if has_fe else 0

    cov, se, zstat, pval, ci = _nb_vcov(
        y=y, X=X, beta=beta, alpha=alpha, mu=mu, dispersion=dispersion,
        fe_groups=fe_groups, has_fe=has_fe, fe_effects=fe_effects, off=off,
        cluster=cluster, cov_type=cov_type, vcov_backend=vcov_backend,
        work=work, fe_parts=fe_parts, k=X.shape[1], n=n, n_absorbed=n_absorbed,
    )

    coefficients = pd.Series(beta, index=design_names)
    std_errors = pd.Series(se, index=design_names)
    z_stats = pd.Series(zstat, index=design_names)
    p_values = pd.Series(pval, index=design_names)
    conf_int = pd.DataFrame({"lower": ci[:, 0], "upper": ci[:, 1]}, index=design_names)

    df_resid = max(n - n_absorbed - k, 1)
    llf = float(res["loglik"])
    deviance = float(_nb_deviance(y, mu, alpha, dispersion))
    pseudo_r2 = float(1.0 - llf / res["ll_null"])

    fitted = pd.Series(mu, index=work.index, name="fitted_mean")

    cov_label = (
        (f"cluster({cluster})" if isinstance(cluster, str)
         else "cluster(" + ", ".join(cluster) + ")")
        if cluster is not None else cov_type
    )

    return NegBinResult(
        formula=formula,
        rhs_formula=rhs_formula,
        nobs=n,
        df_resid=df_resid,
        df_model=X.shape[1],
        cov_type=cov_label,
        dispersion=dispersion,
        coefficients=coefficients,
        std_errors=std_errors,
        z_stats=z_stats,
        p_values=p_values,
        conf_int=conf_int,
        llf=llf,
        deviance=deviance,
        pseudo_r2=pseudo_r2,
        alpha=float(alpha),
        n_absorbed=n_absorbed,
        fixed_effects=fe_parts,
        fitted=fitted,
        call=call,
        vcov_backend=vcov_backend,
        _cov=pd.DataFrame(cov, index=design_names, columns=design_names),
        _x_terms=design_names,
        _work=work,
        _y=y,
        _X=X,
        _fe_groups=fe_groups,
        _fe_effects=fe_effects,
        _off=off,
        _wts=wts,
        _has_fe=has_fe,
    )


# --------------------------------------------------------------------------- #
# Variance / likelihood
# --------------------------------------------------------------------------- #
def _nb_var(mu: np.ndarray, alpha: float, dispersion: str) -> np.ndarray:
    if dispersion == "const":  # NB2
        return mu + alpha * mu ** 2
    return mu * (1.0 + alpha)  # NB1


def _nb_loglik(
    y: np.ndarray, mu: np.ndarray, alpha: float, dispersion: str,
    wts: np.ndarray | None = None,
) -> float:
    from scipy.special import gammaln
    if wts is None:
        wts = np.ones_like(y)
    if dispersion == "const":  # NB2 gamma mixture
        a = 1.0 / alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            term = (
                a * np.log(a / (a + mu))
                + y * np.log(mu / (a + mu))
                + gammaln(y + a)
                - gammaln(a)
                - gammaln(y + 1.0)
            )
        return float(np.sum(wts * term))
    else:  # NB1: Var = mu*(1+alpha) (Hilbe NB1)
        a = 1.0 / alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            term = (
                y * np.log(np.where(mu > 0, mu, 1.0))
                - (y + a) * np.log(mu + alpha)
                + a * np.log(alpha)
                + gammaln(y + a)
                - gammaln(a)
                - gammaln(y + 1.0)
            )
        return float(np.sum(wts * term))


def _nb_deviance(y: np.ndarray, mu: np.ndarray, alpha: float, dispersion: str) -> float:
    from scipy.special import gammaln
    with np.errstate(divide="ignore", invalid="ignore"):
        if dispersion == "const":
            a = 1.0 / alpha
            ll_mod = (
                a * np.log(a / (a + mu))
                + y * np.log(np.where(mu > 0, mu / (a + mu), 1.0))
                + gammaln(y + a) - gammaln(a) - gammaln(y + 1.0)
            )
            ll_sat = np.where(
                y > 0,
                a * np.log(a / (a + y)) + y * np.log(y / (a + y))
                + gammaln(y + a) - gammaln(a) - gammaln(y + 1.0),
                0.0,
            )
            return float(2.0 * np.sum(ll_sat - ll_mod))
        else:
            a = 1.0 / alpha
            ll_mod = (
                y * np.log(np.where(mu > 0, mu, 1.0))
                - (y + a) * np.log(mu + alpha)
                + a * np.log(alpha) + gammaln(y + a) - gammaln(a) - gammaln(y + 1.0)
            )
            ll_sat = np.where(
                y > 0,
                y * np.log(y) - (y + a) * np.log(y + alpha)
                + a * np.log(alpha) + gammaln(y + a) - gammaln(a) - gammaln(y + 1.0),
                0.0,
            )
            return float(2.0 * np.sum(ll_sat - ll_mod))


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def _fit_nb(
    *, y: np.ndarray, X: np.ndarray, fe_groups: list[np.ndarray],
    has_fe: bool, dispersion: str, wts: np.ndarray, off: np.ndarray,
) -> dict[str, Any]:
    """Joint MLE of (beta, alpha). Pooled uses direct scipy optimization
    (exact). FE uses IRLS with alternating-projections FE absorption."""
    if not has_fe:
        return _fit_nb_pooled(y=y, X=X, dispersion=dispersion, wts=wts, off=off)
    return _fit_nb_fe(y=y, X=X, fe_groups=fe_groups, dispersion=dispersion,
                      wts=wts, off=off)


def _neg_ll_joint(params: np.ndarray, y: np.ndarray, X: np.ndarray,
                  dispersion: str, wts: np.ndarray, off: np.ndarray) -> float:
    k = X.shape[1]
    beta = params[:k]
    alpha = np.exp(params[k])
    eta = X @ beta + off
    mu = np.exp(eta)
    return -_nb_loglik(y, mu, alpha, dispersion, wts)


def _fit_nb_pooled(*, y: np.ndarray, X: np.ndarray, dispersion: str,
                   wts: np.ndarray, off: np.ndarray) -> dict[str, Any]:
    k = X.shape[1]
    # Poisson IRLS start for beta
    eta = np.log(np.clip(y, 1, None)) - off
    beta0 = np.zeros(k)
    for _ in range(100):
        mu = np.clip(np.exp(eta), 1e-12, None)
        W = mu
        Xs = X * np.sqrt(W)[:, None]
        zs = (eta + (y - mu) / mu) * np.sqrt(W)
        beta0 = np.linalg.lstsq(Xs, zs, rcond=None)[0]
        eta_new = X @ beta0 + off
        if np.max(np.abs(eta_new - eta)) < 1e-12:
            eta = eta_new
            break
        eta = eta_new
    # method-of-moments start for alpha
    mu0 = np.clip(np.exp(X @ beta0 + off), 1e-12, None)
    resid = y - mu0
    v0 = float(np.var(resid)) if len(y) > 1 else 1.0
    m0 = max(float(np.mean(mu0)), 1e-8)
    if dispersion == "const":
        a0 = max((v0 / m0 - 1.0) / m0, 0.1)
    else:
        a0 = max(v0 / m0 - 1.0, 0.1)
    a0 = float(np.clip(a0, 0.05, 10.0))
    x0 = np.concatenate([beta0, [np.log(a0)]])
    res = minimize(
        _neg_ll_joint, x0, args=(y, X, dispersion, wts, off),
        method="BFGS", options={"gtol": 1e-10, "maxiter": 5000},
    )
    if not res.success or not np.all(np.isfinite(res.x)):
        res = minimize(_neg_ll_joint, np.concatenate([np.zeros(k), [0.0]]),
                       args=(y, X, dispersion, wts, off),
                       method="Nelder-Mead",
                       options={"xatol": 1e-12, "fatol": 1e-12})
    beta = res.x[:k]
    alpha = float(np.exp(res.x[k]))
    eta = X @ beta + off
    mu = np.exp(eta)
    ll = _nb_loglik(y, mu, alpha, dispersion, wts)
    ll_null = _nb_loglik(y, np.full(len(y), max(np.mean(y), 1e-8)),
                         alpha, dispersion, wts)
    return {"beta": beta, "fe_effects": [], "alpha": alpha,
            "loglik": ll, "ll_null": ll_null}


def _fit_nb_fe(*, y: np.ndarray, X: np.ndarray, fe_groups: list[np.ndarray],
               dispersion: str, wts: np.ndarray, off: np.ndarray,
               max_outer: int = 100, tol: float = 1e-10) -> dict[str, Any]:
    """IRLS with N-way FE absorption (alternating projections) + profiled alpha."""
    n, k = X.shape
    beta = np.zeros(k)
    fe_effects = [np.zeros(int(np.unique(g).size)) for g in fe_groups]
    alpha = 1.0

    mu0 = np.full(n, max(np.mean(y), 1e-8))
    ll_null = _nb_loglik(y, mu0, alpha, dispersion, wts)

    for _ in range(max_outer):
        eta = X @ beta + off
        for g, fef in zip(fe_groups, fe_effects):
            eta = eta + fef[g]
        mu = np.clip(np.exp(eta), 1e-12, None)

        alpha = _profile_alpha(y, mu, dispersion, wts, alpha)
        mu = np.clip(np.exp(eta), 1e-12, None)

        V = np.clip(_nb_var(mu, alpha, dispersion), 1e-12, None)
        W = (mu ** 2) / V * wts
        z = eta + (y - mu) / mu

        beta, fe_effects = _wls_fe(z, X, W, fe_groups, beta, fe_effects)

        eta_new = X @ beta + off
        for g, fef in zip(fe_groups, fe_effects):
            eta_new = eta_new + fef[g]
        if np.max(np.abs(eta_new - eta)) < tol:
            break

    eta = X @ beta + off
    for g, fef in zip(fe_groups, fe_effects):
        eta = eta + fef[g]
    mu = np.clip(np.exp(eta), 1e-12, None)
    alpha = _profile_alpha(y, mu, dispersion, wts, alpha)
    mu = np.clip(np.exp(eta), 1e-12, None)
    ll = _nb_loglik(y, mu, alpha, dispersion, wts)
    return {"beta": beta, "fe_effects": fe_effects, "alpha": alpha,
            "loglik": ll, "ll_null": ll_null}


def _profile_alpha(y: np.ndarray, mu: np.ndarray, dispersion: str,
                   wts: np.ndarray, alpha0: float) -> float:
    def neg_ll(a: float) -> float:
        return -_nb_loglik(y, mu, a, dispersion, wts)
    res = minimize_scalar(neg_ll, bounds=(1e-4, 1e4), method="bounded",
                          options={"xatol": 1e-10})
    if res.success and res.x > 0:
        return float(res.x)
    return alpha0


def _wls_fe(z: np.ndarray, X: np.ndarray, W: np.ndarray,
            fe_groups: list[np.ndarray], beta0: np.ndarray,
            fe0: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """One weighted-least-squares step with N-way FE absorbed by iterative
    within-demeaning (Gauss-Seidel alternating projections).  The objective is
    ``sum_i W_i (z_i - x_i'beta - fe_i)^2``; the FE effects are the ``W``-weighted
    group means of the residual after the regressors are projected out."""
    beta = beta0.copy()
    fe_effects = [f.copy() for f in fe0]
    sqrtW = np.sqrt(W)
    Xs = X * sqrtW[:, None]
    zs = z * sqrtW
    for _ in range(500):
        # residual after removing current FE contributions
        resid = zs.copy()
        for gi, g in enumerate(fe_groups):
            resid = resid - fe_effects[gi][g] * sqrtW
        # update beta given FE
        try:
            beta = np.linalg.lstsq(Xs, resid, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = beta0
        # residual after removing regressors (for FE update)
        resid = zs - Xs @ beta
        prev = [f.copy() for f in fe_effects]
        for gi, g in enumerate(fe_groups):
            # subtract the OTHER FE dimensions, keep only dimension gi's resid
            full = resid.copy()
            for gj, gx in enumerate(fe_groups):
                if gj != gi:
                    full = full - fe_effects[gj][gx] * sqrtW
            num = np.bincount(g, weights=(full / np.where(sqrtW > 0, sqrtW, 1.0)) * W,
                              minlength=prev[gi].size)
            den = np.bincount(g, weights=W, minlength=prev[gi].size)
            fe_effects[gi] = np.where(den > 0, num / den, 0.0)
        delta = max(np.max(np.abs(fe_effects[i] - prev[i]))
                    for i in range(len(fe_effects)))
        if delta < 1e-10:
            break
    return beta, fe_effects


def _count_absorbed_dof(df: pd.DataFrame, fe_cols: list[str]) -> int:
    if not fe_cols:
        return 0
    n_groups = sum(df[c].nunique() for c in fe_cols)
    return n_groups - (len(fe_cols) - 1)


def _add_fe(fe_effects: list[np.ndarray], fe_groups: list[np.ndarray]) -> np.ndarray:
    """Sum the per-group FE contributions into a length-n vector."""
    out = np.zeros(len(fe_groups[0]) if fe_groups else 0)
    for fef, g in zip(fe_effects, fe_groups):
        out = out + fef[g]
    return out


# --------------------------------------------------------------------------- #
# Covariance
# --------------------------------------------------------------------------- #
def _nb_vcov(
    *, y: np.ndarray, X: np.ndarray, beta: np.ndarray, alpha: float,
    mu: np.ndarray, dispersion: str, fe_groups: list[np.ndarray], has_fe: bool,
    fe_effects: list[np.ndarray], off: np.ndarray, cluster: Any, cov_type: str,
    vcov_backend: str, work: pd.DataFrame, fe_parts: list[str], k: int, n: int,
    n_absorbed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    V = np.clip(_nb_var(mu, alpha, dispersion), 1e-12, None)
    score_beta = ((y - mu) / V * mu)[:, None] * X  # (n, k)

    W = (mu ** 2) / V * 1.0
    XtX = (X * W[:, None]).T @ X
    try:
        bread = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(XtX)

    if cluster is not None:
        cl = work[cluster if isinstance(cluster, str) else cluster[0]].astype(
            "category").cat.codes.values
        meat = np.zeros((k, k))
        for c in np.unique(cl):
            gg = score_beta[cl == c].sum(axis=0)
            meat += np.outer(gg, gg)
        cov = bread @ meat @ bread
        if vcov_backend == "stata":
            G = len(np.unique(cl))
            cov = cov * (G / (G - 1))
        else:
            cov = cov * ((n - 1) / (n - k))
    else:
        if cov_type == "nonrobust":
            cov = bread
        else:
            if cov_type == "HC0":
                w = np.ones(n)
            elif cov_type == "HC1":
                w = np.full(n, n / (n - k))
            elif cov_type in ("HC2", "HC3"):
                h = np.einsum("ni,ij,nj->n", X, bread, X) * W
                w = 1.0 / (1.0 - h) if cov_type == "HC2" else 1.0 / (1.0 - h) ** 2
            else:
                w = np.ones(n)
            meat = (score_beta * w[:, None]).T @ score_beta
            cov = bread @ meat @ bread

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    z = beta / np.where(se > 0, se, np.nan)
    p = 2.0 * (1.0 - _norm_cdf(np.abs(z)))
    crit = _norm_ppf(0.975)
    ci = np.column_stack([beta - crit * se, beta + crit * se])
    return cov, se, z, p, ci


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    from scipy import stats as _st
    return _st.norm.cdf(x)


def _norm_ppf(q: float) -> float:
    from scipy import stats as _st
    return float(_st.norm.ppf(q))
