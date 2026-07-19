"""Pure diagnostic functions for OLS results.

Each function operates on residuals and the design matrix ``X`` (with the
intercept column included, matching Stata's ``estat`` family conventions).
They are deliberately backend-agnostic so they work whether the parent
:class:`~open_econs.core.results.OLSResult` was produced by statsmodels or by
open-econs' own OLS engine (which stores ``_fit=None``).

Convention notes (parity with Stata ``estat`` / R):

* **Breusch-Godfrey** (``bg``): the auxiliary regression includes the FULL
  design matrix (constant + regressors) and ``lags`` lagged residuals, exactly
  as Stata ``estat bgodfrey``. The LM statistic is ``n * R2`` ~ chi2(lags).
  Stata additionally reports an F version; statsmodels'
  ``acorr_breusch_godfrey`` returns both.
* **White**: ``estat imtest, white`` regresses resid^2 on X, squares, and
  pairwise cross-products; LM = n*R2 ~ chi2(df).
* **Cook's distance / leverage / DFBETAS**: computed from the hat matrix
  ``H = X (X'X)^{-1} X'`` and studentized residuals, matching Stata
  ``predict, cooksd`` / ``predict, leverage`` and R ``cooks.distance`` /
  ``hatvalues`` / ``dfbetas``. Stata ``dfbeta`` returns the RAW DFBETA; the
  standardized ``DFBETAS`` (divided by the leave-one-out SE) is what R
  ``dfbetas`` and Stata ``predict, dfbeta``-style standardization produce. We
  return DFBETAS (standardized) as the primary ``dfbetas()`` method and expose
  ``dfbeta()`` (raw) for Stata-equivalence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _stats
from statsmodels.stats.diagnostic import acorr_ljungbox as _sm_ljungbox


def _as_float_matrix(X: object) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.values.astype(float)
    return np.asarray(X, dtype=float)


def _hat_matrix_diag(X: np.ndarray) -> np.ndarray:
    """Diagonal of the hat matrix H = X (X'X)^{-1} X'."""
    XtX_inv = np.linalg.inv(X.T @ X)
    H = X @ XtX_inv @ X.T
    return np.diag(H)


def _studentized_residuals(
    resid: np.ndarray, X: np.ndarray, *, external: bool = True
) -> np.ndarray:
    """Internally (external=False) or externally (external=True) studentized.

    Externally studentized uses the leave-one-out residual variance
    ``s_{(-i)}^2``; matches Stata/R ``rstudent``.
    """
    Xv = _as_float_matrix(X)
    res = np.asarray(resid, dtype=float).ravel()
    n, k = Xv.shape
    hat = _hat_matrix_diag(Xv)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = res / np.sqrt(1.0 - hat)
    if not external:
        s2 = np.sum(res ** 2) / (n - k)
        return t / np.sqrt(s2)
    out = np.empty(n, dtype=float)
    s2 = np.sum(res ** 2) / (n - k)
    for i in range(n):
        se_i = np.sqrt(max(s2 * (1.0 - hat[i]), 0.0))
        out[i] = t[i] / se_i if se_i > 0 else float("nan")
    return out


def breusch_godfrey(
    resid: np.ndarray, X: np.ndarray, lags: int = 1,
) -> dict[str, float]:
    """Breusch-Godfrey LM test for residual autocorrelation.

    ``X`` is the FULL design matrix (constant + regressors), matching Stata's
    ``estat bgodfrey``. The auxiliary regression is of ``resid`` on ``X`` and
    ``lags`` lagged residuals; the LM statistic is ``n * R2`` ~ chi2(lags). An
    F version (Stata reports it) uses the standard F-test of the lag
    coefficients.

    NOTE: statsmodels 0.14's ``acorr_breusch_godfrey`` no longer accepts the
    design matrix (it runs the auxiliary regression on lagged residuals only,
    WITHOUT the regressors), so it does NOT match Stata. We implement the test
    from scratch instead.
    """
    Xv = _as_float_matrix(X)
    res = np.asarray(resid, dtype=float).ravel()
    n = len(res)
    L = np.zeros((n, lags))
    for j in range(1, lags + 1):
        col = np.zeros(n)
        col[j:] = res[: n - j]
        L[:, j - 1] = col
    Z = np.column_stack([Xv, L])
    beta = np.linalg.lstsq(Z, res, rcond=None)[0]
    fitted = Z @ beta
    ssr = float(np.sum((res - fitted) ** 2))
    ssr0 = float(np.sum((res - res.mean()) ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        lm = n * (ssr0 - ssr) / ssr0 if ssr0 > 0 else float("nan")
    p_lm = float(_stats.chi2.sf(lm, lags))
    # F version: test the `lags` lag coefficients jointly
    q = lags
    df_den = n - Z.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f_stat = ((ssr0 - ssr) / q) / (ssr / df_den) if ssr > 0 and df_den > 0 else float("nan")
    p_f = float(_stats.f.sf(f_stat, q, df_den)) if df_den > 0 else float("nan")
    return {
        "lm_stat": float(lm),
        "lm_pvalue": p_lm,
        "f_stat": float(f_stat),
        "f_pvalue": p_f,
        "df": float(lags),
    }


def white_heteroskedasticity(
    resid: np.ndarray, X: np.ndarray, interaction: bool = True,
) -> dict[str, float]:
    """White's general heteroskedasticity test.

    Auxiliary OLS of resid^2 on the regressors (EXCLUDING the constant),
    their squares, and (if ``interaction``) pairwise cross-products — each
    term mean-centered, matching Stata ``estat imtest, white``. The LM
    statistic is ``n * R2`` ~ chi2(df) with df = p + p(p+1)/2 for ``p``
    non-constant regressors.
    """
    Xv = _as_float_matrix(X)
    n, k = Xv.shape
    u2 = np.asarray(resid, dtype=float).ravel() ** 2

    # Identify non-constant columns (Stata builds White terms from the
    # regressors only, not the constant).
    non_const = [j for j in range(k) if np.std(Xv[:, j]) > 1e-12]
    Xc = Xv[:, non_const] if non_const else np.zeros((n, 0))
    p = Xc.shape[1]

    cols: list[np.ndarray] = [Xc[:, j] for j in range(p)]
    cols += [Xc[:, j] ** 2 for j in range(p)]
    if interaction:
        for a in range(p):
            for b in range(a + 1, p):
                cols.append(Xc[:, a] * Xc[:, b])

    Z = np.column_stack(cols) if cols else np.zeros((n, 0))
    # Stata centers the terms in `imtest, white`; the auxiliary constant
    # absorbs the grand mean so we center each term.
    Z = Z - Z.mean(axis=0)
    Z = np.column_stack([np.ones(n), Z])

    beta = np.linalg.lstsq(Z, u2, rcond=None)[0]
    fitted = Z @ beta
    ssr = float(np.sum((u2 - fitted) ** 2))
    ssr0 = float(np.sum((u2 - u2.mean()) ** 2))
    df = Z.shape[1] - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        lm = n * (ssr0 - ssr) / ssr0 if ssr0 > 0 else float("nan")
    pval = float(_stats.chi2.sf(lm, df)) if df > 0 else float("nan")
    return {"white_stat": float(lm), "white_pvalue": pval, "df": float(df)}


def ljung_box(
    resid: np.ndarray, lags: int = 1, box_pierce: bool = False,
) -> dict[str, float]:
    """Ljung-Box Q test on residuals (mean ~ 0 by construction)."""
    res = np.asarray(resid, dtype=float).ravel()
    out = _sm_ljungbox(res, lags=[lags], boxpierce=box_pierce, return_df=True)
    result: dict[str, float] = {
        "lb_stat": float(out["lb_stat"].iloc[0]),
        "lb_pvalue": float(out["lb_pvalue"].iloc[0]),
    }
    if box_pierce:
        result["bp_stat"] = float(out["bp_stat"].iloc[0])
        result["bp_pvalue"] = float(out["bp_pvalue"].iloc[0])
    return result


def cooks_distance(resid: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cook's distance per observation.

    ``D_i = (t_i^2 / k) * (h_ii / (1 - h_ii))`` with ``t_i`` the (internally)
    studentized residual and ``h_ii`` leverage. Matches Stata
    ``predict, cooksd``.
    """
    Xv = _as_float_matrix(X)
    res = np.asarray(resid, dtype=float).ravel()
    n, k = Xv.shape
    hat = _hat_matrix_diag(Xv)
    s2 = np.sum(res ** 2) / (n - k)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (res / np.sqrt(1.0 - hat)) / np.sqrt(s2)
        cooks = (t ** 2 / k) * (hat / (1.0 - hat))
    return cooks


def leverage(X: np.ndarray) -> np.ndarray:
    """Leverage = diagonal of the hat matrix."""
    return _hat_matrix_diag(_as_float_matrix(X))


def dfbetas(
    X: np.ndarray, coefficients: np.ndarray, resid: np.ndarray,
) -> np.ndarray:
    """Standardized DFBETAS, an (n, k) array indexed [obs, param].

    ``DFBETAS_{ij} = (b_j - b_{j(-i)}) / SE_j(-i)`` using the leave-one-out
    coefficient and standard error. Matches R ``dfbetas`` / Stata
    ``predict, dfbeta`` standardization.
    """
    Xv = _as_float_matrix(X)
    b = np.asarray(coefficients, dtype=float).ravel()
    res = np.asarray(resid, dtype=float).ravel()
    n, k = Xv.shape
    XtX_inv = np.linalg.inv(Xv.T @ Xv)
    hat = np.diag(Xv @ XtX_inv @ Xv.T)

    out = np.empty((n, k), dtype=float)
    XtX_inv_Xt = XtX_inv @ Xv.T
    for i in range(n):
        h_i = hat[i]
        if h_i >= 1.0:
            out[i, :] = np.nan
            continue
        # leave-one-out coefficient: b(-i) = b - (1/(1-h_i)) * (X'X)^-1 X_i' e_i
        e_i = res[i]
        delta = XtX_inv_Xt[:, i] * (e_i / (1.0 - h_i))
        b_minus = b - delta
        # leave-one-out residual variance
        s2_i = (np.sum(res ** 2) - (res[i] ** 2) / (1.0 - h_i)) / (n - k - 1)
        se_j = np.sqrt(np.maximum(s2_i, 0.0)) * np.sqrt(np.diag(XtX_inv))
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i, :] = (b - b_minus) / se_j
    return out


def dfbeta(
    X: np.ndarray, coefficients: np.ndarray, resid: np.ndarray,
) -> np.ndarray:
    """Raw DFBETA = b_j - b_{j(-i)}, an (n, k) array. Matches Stata ``dfbeta``.

    Stata's ``dfbeta`` command drops the constant by default; the array here
    includes all parameters (including the constant) so callers can slice.
    """
    Xv = _as_float_matrix(X)
    res = np.asarray(resid, dtype=float).ravel()
    n, k = Xv.shape
    XtX_inv = np.linalg.inv(Xv.T @ Xv)
    hat = np.diag(Xv @ XtX_inv @ Xv.T)
    XtX_inv_Xt = XtX_inv @ Xv.T
    out = np.empty((n, k), dtype=float)
    for i in range(n):
        h_i = hat[i]
        if h_i >= 1.0:
            out[i, :] = np.nan
            continue
        e_i = res[i]
        delta = XtX_inv_Xt[:, i] * (e_i / (1.0 - h_i))
        out[i, :] = delta
    return out
