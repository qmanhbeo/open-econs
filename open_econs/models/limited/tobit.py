"""Tobit (censored normal) MLE estimator.

Backend: a hand-rolled censored-normal maximum-likelihood estimator
(``scipy.optimize.minimize`` over the Tobit log-likelihood). ``statsmodels``
0.14.6 has **no** Tobit model, so — unlike ``ologit``/``oprobit`` — we cannot
wrap a statsmodels routine; the censored likelihood, its gradient, and the
OIM covariance must be derived directly (see ``methodology/limited/tobit.md``).

Parity target (rule 1): Stata base ``tobit`` (left-censored ``ll(0)`` by
default, optional ``ul(.)``) and R ``AER::tobit`` (``left=0, right=Inf``).
The point estimates, ``sigma`` (Stata prints ``Log(scale)`` = ``log sigma``),
and the OIM (nonrobust) covariance match Stata/R to ``1e-6``. The high-precision
BFGS polish pass (gtol=1e-12, ftol=1e-14) is what lands the coefficients at
1e-6; the raw ``optimize.minimize`` default stops ~1e-5 short of Stata.

Supports ``formula`` + ``data``, ``cov_type`` in
``{nonrobust, HC0, HC1, HC2, HC3, cluster}``, and censoring limits
``left`` / ``right``. No fixed-effects arguments: base Stata ``tobit`` and
R ``AER::tobit`` have none.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from open_econs.core.call_capture import capture_call as _capture_call
from open_econs._internal import errors
from open_econs.core.results import TobitResult
from open_econs.core.cov_type import validate_cov_type


def tobit(
    formula: str,
    data: pd.DataFrame,
    left: float | None = 0.0,
    right: float | None = None,
    cov_type: str = "nonrobust",
    cluster: str | list[str] | None = None,
) -> TobitResult:
    """Estimate a Tobit (censored normal) regression by maximum likelihood.

    The latent variable is ``y* = x'b + u``, ``u ~ N(0, sigma^2)``, observed as
    ``y = max(left, min(right, y*))``. Left-censoring at ``left`` (Stata default
    ``ll(0)``) and right-censoring at ``right`` (Stata ``ul(.)``; default
    ``None`` = no right censoring) are both supported.

    Parameters
    ----------
    formula : str
        Two-sided formula, e.g. ``"y ~ x1 + x2"``.
    data : pd.DataFrame
        Data containing all formula variables.
    left : float, optional
        Left-censoring limit. ``0.0`` (default) reproduces Stata
        ``tobit y x, ll(0)`` and R ``AER::tobit(y ~ x, left = 0)``. Pass
        ``None`` for no left censoring.
    right : float, None, optional
        Right-censoring limit. ``None`` (default) = no right censoring
        (Stata ``tobit`` with no ``ul()``); ``Inf`` in R ``AER::tobit`` is
        the same as ``None`` here.
    cov_type : str, default "nonrobust"
        Covariance estimator for the **regression coefficients** (the ``sigma``
        row is dropped before reporting ``coef``/``SE``). One of
        ``"nonrobust"`` (OIM), ``"HC0"``, ``"HC1"``, ``"HC2"``, ``"HC3"``.
        Ignored when ``cluster`` is given.
    cluster : str or list of str, optional
        Column(s) for cluster-robust (CRV1) standard errors; a list requests
        multi-way clustering. Takes precedence over *cov_type*.

    Returns
    -------
    TobitResult
        Immutable result with ``.coefficients`` (regressors only, excluding
        ``sigma``), ``.sigma``, ``.tidy()``, ``.summary()``,
        ``.predict(type="ystar"|"y"|"cond"|"pr_gt0")``, and ``.margins()``.
        Stata prints ``Log(scale)`` = ``log(sigma)`` — see ``TobitResult``.

    Examples
    --------
    >>> import open_econs as oe
    >>> r = oe.tobit("y ~ x1 + x2", data=df, left=0)
    >>> r.tidy(); r.sigma; r.summary()
    >>> r.predict(type="y")          # E[y | x]  (observed, censored-aware)
    >>> r.predict(type="ystar")      # E[y* | x] (latent)
    >>> r.predict(type="pr_gt0")     # P(y > 0 | x)
    """
    call = _capture_call(
        formula=formula, left=left, right=right, cov_type=cov_type, cluster=cluster
    )

    if cluster is not None:
        cov_type = "cluster"
        if isinstance(cluster, str):
            cluster_cols = [cluster]
        else:
            cluster_cols = list(cluster)
        cov_label = "cluster(" + (cluster if isinstance(cluster, str) else ", ".join(cluster)) + ")"
    else:
        cov_type = validate_cov_type(
            cov_type,
            accepted={"nonrobust", "HC0", "HC1", "HC2", "HC3"},
            estimator="tobit()",
        )
        cluster_cols = []
        cov_label = cov_type

    rhs_formula = formula.split("~", 1)[1].strip()

    from formulaic import Formula
    try:
        formula_obj = Formula(formula)
        model_spec = formula_obj.get_model_matrix(data, na_action="drop")
    except Exception as e:
        msg = str(e)
        if "not present in the dataset" in msg or "is not present" in msg:
            import re as _re
            m = _re.search(r"`(\w+)`", msg)
            bad_col = m.group(1) if m else formula.split("~", 1)[1].strip()
            raise errors.missing_column_error(bad_col, data.columns.tolist()) from e
        raise

    if hasattr(model_spec, "rhs"):
        XX = model_spec.rhs
        yy = model_spec.lhs
    else:
        from open_econs._internal.formula import parse_formula as _parse
        yy, XX = _parse(formula, data)

    # Keep the intercept column (Tobit has a free constant); do NOT drop it.
    original_n = len(data)
    dropped = original_n - len(yy)
    if dropped > 0:
        warnings.warn(
            errors.rows_dropped_warning(dropped, original_n, []),
            RuntimeWarning,
            stacklevel=3,
        )
    if len(yy) == 0:
        raise errors.empty_data_error(original_n, dropped, [])

    y_series = yy.iloc[:, 0] if hasattr(yy, "iloc") else pd.Series(yy)
    y = pd.Series(y_series).reset_index(drop=True).astype(float)
    X = XX.reset_index(drop=True).astype(float)

    if cluster_cols:
        for c in cluster_cols:
            if c not in data.columns:
                raise errors.missing_column_error(c, data.columns.tolist())
        cluster_data = data[cluster_cols].reset_index(drop=True)
    else:
        cluster_data = None

    # Resolve censoring limits.
    left_lim = float(left) if left is not None else -np.inf
    right_lim = float(right) if right is not None else np.inf

    # ---- MLE ----
    names = [str(c) for c in X.columns]
    k = X.shape[1]
    Xv = X.values
    n = Xv.shape[0]

    # Starting values: OLS on observed y.
    try:
        beta0_ols, _, _, _ = np.linalg.lstsq(Xv, y.values, rcond=None)
    except Exception:
        beta0_ols = np.zeros(k)
    resid = y.values - Xv @ beta0_ols
    sigma0 = float(np.sqrt(np.sum(resid ** 2) / max(n, 1)))
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = float(np.std(y.values)) + 1e-3
    start = np.concatenate([beta0_ols, [np.log(sigma0)]])

    def neg_loglik(p: np.ndarray) -> float:
        beta = p[:k]
        lsig = p[k]
        sig = np.exp(lsig)
        xb = Xv @ beta
        out = _tobit_loglik(y.values, xb, sig, left_lim, right_lim, n)
        return -out

    # 1) Stata-like default (Newton-Raphson via BFGS) for a stable start.
    fit1 = minimize(
        neg_loglik, start, method="BFGS",
        options={"gtol": 1e-8, "maxiter": 2000},
    )
    # 2) High-precision polish so point estimates / sigma / OIM SEs match
    #    Stata & R to 1e-6. The raw optimizer stops ~1e-5 short (see methodology).
    polished = minimize(
        neg_loglik, fit1.x, method="L-BFGS-B",
        options={"gtol": 1e-12, "ftol": 1e-14, "maxiter": 10000},
    )
    params = polished.x
    beta = params[:k]
    lsig = params[k]
    sigma = float(np.exp(lsig))

    # ---- Covariance ----
    # OIM (nonrobust) covariance = inv(Hessian of the total NLL) on the
    # (beta, sigma) parameterization. This matches Stata/R OIM to 1e-6.
    # Robust/cluster use the analytic-score sandwich: OIM bread (the same
    # inv(Hessian) above) multiplied by the meat = sum of per-observation
    # score outer-products, where the per-observation scores are the EXACT
    # analytic Tobit score contributions (including the censored/limit-of-
    # observation regions), evaluated on the (beta, sigma) parameterization to
    # match the OIM bread. This is Stata's exact OIM-robust bread and closes
    # the ~1e-4 gap left by the old numerical (beta, ln sigma) sandwich.
    bread_full = _oim_cov(y.values, Xv, beta, sigma, left_lim, right_lim)

    if cov_type == "nonrobust":
        cov_full = bread_full
    else:
        cov_full = _sandwich_cov(
            y.values, Xv, beta, sigma, left_lim, right_lim,
            cov_type, cluster_data, cluster_cols, bread_full,
        )

    # Report regressor rows only (drop the sigma row) for coef/SE.
    coef_arr = beta
    se_arr = np.sqrt(np.diag(cov_full)[:k])
    z_arr = coef_arr / se_arr
    p_arr = 2.0 * norm.sf(np.abs(z_arr))
    crit = norm.ppf(0.975)
    conf_arr = np.column_stack([
        coef_arr - crit * se_arr,
        coef_arr + crit * se_arr,
    ])

    coefficients = pd.Series(coef_arr, index=names)
    std_errors = pd.Series(se_arr, index=names)
    conf_int = pd.DataFrame(
        {"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]}, index=names
    )

    llf = float(-neg_loglik(params))

    # Stata reports the count of left- and right-censored observations.
    n_left = int(np.sum(y.values <= left_lim + 1e-12)) if np.isfinite(left_lim) else 0
    n_right = int(np.sum(y.values >= right_lim - 1e-12)) if np.isfinite(right_lim) else 0

    # Fitted quantities (in-sample, matching predict()).
    xb = Xv @ beta
    Phi_L = norm.cdf((left_lim - xb) / sigma) if np.isfinite(left_lim) else np.zeros(n)
    lam_L = _inv_mills((left_lim - xb) / sigma)
    lam_R = _inv_mills((right_lim - xb) / sigma)
    # E[y* | x] = xb
    # E[y | x]  = xb + sigma*(lam_L - lam_R)  (within [left, right])
    e_y = xb + sigma * (lam_L - lam_R)
    # P(y > left) = 1 - Phi_L  (probability of being uncensored on the left)
    pr_gt0 = 1.0 - Phi_L

    fitted_ystar = pd.Series(xb, index=X.index, name="fitted_ystar")
    fitted_y = pd.Series(e_y, index=X.index, name="fitted_y")
    fitted_pr = pd.Series(pr_gt0, index=X.index, name="pr_gt0")

    df_model = k
    df_resid = n - k - 1

    result = TobitResult(
        formula=formula,
        rhs_formula=rhs_formula,
        nobs=int(n),
        df_resid=int(df_resid),
        df_model=int(df_model),
        cov_type=cov_label,
        coefficients=coefficients,
        std_errors=std_errors,
        z_stats=pd.Series(z_arr, index=names),
        p_values=pd.Series(p_arr, index=names),
        conf_int=conf_int,
        sigma=sigma,
        log_scale=lsig,
        llf=llf,
        n_left=n_left,
        n_right=n_right,
        left=left_lim,
        right=right_lim,
        fitted_ystar=fitted_ystar,
        fitted_y=fitted_y,
        fitted_pr=fitted_pr,
        call=call,
        _cov=pd.DataFrame(
            cov_full,
            index=names + ["sigma"],
            columns=names + ["sigma"],
        ),
        _params=params,
        _X=Xv,
        _y=y.values,
        _names=names,
    )
    return result


def _tobit_loglik(
    y: np.ndarray,
    xb: np.ndarray,
    sigma: float,
    left_lim: float,
    right_lim: float,
    n: int,
) -> float:
    s = 0.0
    if np.isfinite(left_lim):
        left_idx = y <= left_lim + 1e-12
        if np.any(left_idx):
            s += np.sum(
                np.log(norm.cdf((left_lim - xb[left_idx]) / sigma) + 1e-300)
            )
    if np.isfinite(right_lim):
        right_idx = y >= right_lim - 1e-12
        if np.any(right_idx):
            s += np.sum(
                np.log(1.0 - norm.cdf((right_lim - xb[right_idx]) / sigma) + 1e-300)
            )
    mid_idx: np.ndarray = np.ones(n, dtype=bool)
    if np.isfinite(left_lim):
        mid_idx = mid_idx & (y > left_lim + 1e-12)
    if np.isfinite(right_lim):
        mid_idx = mid_idx & (y < right_lim - 1e-12)
    if np.any(mid_idx):
        s += np.sum(
            np.log(norm.pdf((y[mid_idx] - xb[mid_idx]) / sigma) / sigma + 1e-300)
        )
    return s


def _inv_mills(z: np.ndarray) -> np.ndarray:
    # E[u | u > -z] density ratio for truncated normal; returns 0 where z is
    # extreme (avoids 0/0). lambda(z) = phi(z) / (1 - Phi(z)).
    z = np.asarray(z, dtype=float)
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    denom = np.clip(1.0 - cdf, 1e-300, None)
    return pdf / denom


def _oim_cov(
    y: np.ndarray,
    Xv: np.ndarray,
    beta: np.ndarray,
    sigma: float,
    left_lim: float,
    right_lim: float,
) -> np.ndarray:
    """OIM (nonrobust) covariance = inv(-Hessian) of the full log-likelihood.

    Parameterization is ``(beta, sigma)`` (NOT ln sigma — the sigma scale ~1 is
    numerically well-conditioned at the optimum, unlike the log-transform which
    makes a flat, ill-conditioned direction that breaks numeric Hessians).
    Stata ``tobit`` OIM is exactly ``-Hessian``; we evaluate it with
    ``statsmodels.tools.numdiff.approx_hess`` (2nd-order central differences on
    the total negative log-likelihood), which matches Stata/R to 1e-6.
    """
    from statsmodels.tools.numdiff import approx_hess

    p0 = np.concatenate([beta, [sigma]])

    def _nll(pvec: np.ndarray) -> float:
        b = pvec[: Xv.shape[1]]
        sig = float(pvec[Xv.shape[1]])
        xb = Xv @ b
        return -_tobit_loglik(y, xb, sig, left_lim, right_lim, len(y))

    hess = approx_hess(p0, _nll)
    # _nll is the NEGATIVE log-likelihood, so its Hessian is the (positive
    # definite) observed information; the OIM covariance is its inverse.
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hess)
    return cov


def _sandwich_cov(
    y: np.ndarray,
    Xv: np.ndarray,
    beta: np.ndarray,
    sigma: float,
    left_lim: float,
    right_lim: float,
    cov_type: str,
    cluster_data: "pd.DataFrame | None",
    cluster_cols: list[str],
    bread: np.ndarray,
) -> np.ndarray:
    """HC0/HC1/HC2/HC3 and cluster-robust covariance via analytic scores.

    The per-observation scores are the EXACT analytic Tobit score
    contributions (including the censored/limit-of-observation regions),
    evaluated on the ``(beta, sigma)`` parameterization to match the OIM
    ``bread`` (``inv(Hessian)``). This is Stata's exact OIM-robust bread and
    closes the ~1e-4 gap left by the old numerical (beta, ln sigma) sandwich.

    ``bread`` is the OIM covariance (``inv(Hessian)`` on (beta, sigma)),
    passed in so the robust/cluster sandwich shares the same bread Stata
    uses for ``vce(robust)`` / ``vce(cluster)``.
    """
    scores = _tobit_scores(y, Xv, beta, sigma, left_lim, right_lim)
    scores = np.asarray(scores, dtype=float)
    n, q = scores.shape

    if cov_type == "cluster":
        # CRV1: sum meat over clusters, bread = OIM (inv(Hessian)).
        # Stata's ``tobit, vce(cluster ...)`` applies the cluster bias
        # correction G/(G-1) (verified to 1e-10 against Stata).
        assert cluster_data is not None
        meat = np.zeros((q, q))
        n_groups = 0
        if len(cluster_cols) == 1:
            groups = cluster_data[cluster_cols[0]].values
            uniq = pd.unique(groups)
            n_groups = len(uniq)
            for g in uniq:
                gi = groups == g
                sg = scores[gi].sum(axis=0)
                meat += np.outer(sg, sg)
        else:
            for c in cluster_cols:
                groups = cluster_data[c].values
                uniq = pd.unique(groups)
                n_groups += len(uniq)
                for g in uniq:
                    gi = groups == g
                    sg = scores[gi].sum(axis=0)
                    meat += np.outer(sg, sg)
        meat *= n_groups / (n_groups - 1.0)
        return bread @ meat @ bread
    else:
        # Stata's ``tobit, vce(robust)`` is the OIM bread with the OPG meat
        # scaled by the small-sample correction n/(n-1) (verified to 1e-10;
        # note this is NOT the usual n/(n-q) HC1 factor). For tobit the
        # canonical robust SE users expect IS Stata's vce(robust), so OE exposes
        # it as ``HC1`` (and ``HC0`` is the uncorrected OPG meat). HC2/HC3 are
        # the leverage-weighted variants (no Stata equivalent for tobit).
        opg_inv = np.linalg.inv(scores.T @ scores)
        if cov_type == "HC0":
            w = np.ones(n)
        elif cov_type == "HC1":
            w = np.full(n, n / (n - 1.0))
        elif cov_type in ("HC2", "HC3"):
            h = np.einsum("ni,ij,jn->n", scores, opg_inv, scores.T)
            if cov_type == "HC2":
                w = 1.0 / (1.0 - h)
            else:
                w = 1.0 / (1.0 - h) ** 2
        else:
            raise ValueError(f"Unsupported cov_type for sandwich: {cov_type}")
        meat = (scores * w[:, None]).T @ scores
        return bread @ meat @ bread


def _tobit_scores(
    y: np.ndarray,
    Xv: np.ndarray,
    beta: np.ndarray,
    sigma: float,
    left_lim: float,
    right_lim: float,
) -> np.ndarray:
    """Exact per-observation Tobit score contributions on (beta, sigma).

    For each observation i (x_i = row of X, xb_i = x_i'b, z_i = (y_i - xb_i)/σ):

      * uncensored:  s_b = z_i x_i / σ,                       s_σ = (z_i² - 1)/σ
      * left-censored (y_i = L):  λ = φ(w)/Φ(w), w = (L - xb_i)/σ
                                  s_b = -λ x_i / σ,           s_σ = -w λ / σ
      * right-censored (y_i = R): λ = φ(w)/(1-Φ(w)), w = (R - xb_i)/σ
                                  s_b = +λ x_i / σ,           s_σ = +w λ / σ

    These are the analytic score contributions Stata's ``tobit, vce(...)``
    uses; the bread of the robust/cluster sandwich is the OIM (inv(Hessian)).
    """
    n, k = Xv.shape
    xb = Xv @ beta
    scores = np.zeros((n, k + 1))

    mid = np.ones(n, dtype=bool)
    if np.isfinite(left_lim):
        left_idx = y <= left_lim + 1e-12
        mid = mid & (~left_idx)
    else:
        left_idx = np.zeros(n, dtype=bool)
    if np.isfinite(right_lim):
        right_idx = y >= right_lim - 1e-12
        mid = mid & (~right_idx)
    else:
        right_idx = np.zeros(n, dtype=bool)

    if np.any(mid):
        z = (y[mid] - xb[mid]) / sigma
        scores[mid, :k] = (Xv[mid].T * (z / sigma)).T
        scores[mid, k] = (z ** 2 - 1.0) / sigma

    if np.any(left_idx):
        w = (left_lim - xb[left_idx]) / sigma
        pdf = norm.pdf(w)
        cdf = norm.cdf(w)
        lam = pdf / np.clip(cdf, 1e-300, None)
        scores[left_idx, :k] = (Xv[left_idx].T * (-lam / sigma)).T
        scores[left_idx, k] = (-w * lam) / sigma

    if np.any(right_idx):
        w = (right_lim - xb[right_idx]) / sigma
        pdf = norm.pdf(w)
        cdf = norm.cdf(w)
        lam = pdf / np.clip(1.0 - cdf, 1e-300, None)
        scores[right_idx, :k] = (Xv[right_idx].T * (lam / sigma)).T
        scores[right_idx, k] = (w * lam) / sigma

    return scores


def _tobit_loglik_obs(
    y: np.ndarray,
    xb: np.ndarray,
    sigma: float,
    left_lim: float,
    right_lim: float,
) -> np.ndarray:
    n = len(y)
    s = np.zeros(n)
    if np.isfinite(left_lim):
        idx = y <= left_lim + 1e-12
        s[idx] = np.log(norm.cdf((left_lim - xb[idx]) / sigma) + 1e-300)
    if np.isfinite(right_lim):
        idx = y >= right_lim - 1e-12
        s[idx] = np.log(1.0 - norm.cdf((right_lim - xb[idx]) / sigma) + 1e-300)
    mid: np.ndarray = np.ones(n, dtype=bool)
    if np.isfinite(left_lim):
        mid = mid & (y > left_lim + 1e-12)
    if np.isfinite(right_lim):
        mid = mid & (y < right_lim - 1e-12)
    s[mid] = np.log(norm.pdf((y[mid] - xb[mid]) / sigma) / sigma + 1e-300)
    return s
