"""VAR/VECM/Johansen wrapper around ``statsmodels.tsa.vector_ar``.

Provides Stata/R-parity wrappers for:
- ``var_fit()`` -- estimate a VAR(p) model
- ``var_select_order()`` -- lag-order selection with dual IC conventions
- ``johansen_cointegration()`` -- Johansen test with Osterwald-Lenum CVs
- ``granger_causality()`` -- Granger / instantaneous causality tests
- ``vec2var()`` -- VECM-to-VAR conversion for post-cointegration causality

Conventions matched:
- Johansen CVs: Osterwald-Lenum (1992) as default (matching Stata ``vecrank``
  and R ``urca::ca.jo``), overriding statsmodels' MacKinnon (1996).
- VAR IC: standard mode includes all parameters in penalty (matching Stata
  ``varsoc`` / R ``VARselect`` / statsmodels); ``lutstats`` mode excludes
  deterministic terms (matching Stata ``lutstats`` option).
- Granger: F-test default (matching R ``vars`` / statsmodels), chi-squared
  exposed for Stata parity.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

from open_econs.core.call_capture import capture_call as _capture_call
from open_econs.models.timeseries.results import (
    GrangerResult,
    JohansenResult,
    LagOrderResult,
    VARResult,
    VECMResult,
)

from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import (
    VECM,
    coint_johansen,
)


# ── Osterwald-Lenum (1992) critical value tables ─────────────────
# Source: Stata 17 ``_vecgetcv.ado`` lines 1-93, and R ``urca`` ca.jo
# reference: Osterwald-Lenum, Oxford Bull Econ Stats 54(3) 1992, pp. 461-472.
# Tables are (11 x 3) for n=1..11 variables, columns = 10%, 5%, 1%.
# _OL_TRACE[col_idx][n-1] gives the CV for the given significance level.
# col_idx: 0=10%, 1=5%, 2=1%

# Trace statistic critical values (Osterwald-Lenum 1992, Table 1)
_OL_TRACE: dict[str, dict[int, float]] = {
    "none": {  # det_order = -1, no trend
        1: 3.84, 2: 12.53, 3: 24.31, 4: 39.89, 5: 59.46,
        6: 82.49, 7: 109.99, 8: 141.20, 9: 175.77, 10: 212.67, 11: 255.27,
    },
    "const": {  # det_order = 0, unrestricted constant
        1: 3.76, 2: 15.41, 3: 29.68, 4: 47.21, 5: 68.52,
        6: 94.15, 7: 124.24, 8: 156.00, 9: 192.89, 10: 233.13, 11: 277.71,
    },
    "trend": {  # det_order = 1, unrestricted trend
        1: 3.74, 2: 18.17, 3: 34.55, 4: 54.64, 5: 77.74,
        6: 104.94, 7: 136.61, 8: 170.80, 9: 208.97, 10: 250.84, 11: 295.99,
    },
}

# Max-eigenvalue statistic critical values (Osterwald-Lenum 1992, Table 2)
_OL_MAXEIG: dict[str, dict[int, float]] = {
    "none": {  # det_order = -1
        1: 3.84, 2: 11.44, 3: 17.89, 4: 23.80, 5: 30.04,
        6: 36.36, 7: 41.51, 8: 47.99, 9: 53.69, 10: 59.06, 11: 65.30,
    },
    "const": {  # det_order = 0
        1: 3.76, 2: 14.07, 3: 20.97, 4: 27.07, 5: 33.46,
        6: 39.37, 7: 45.28, 8: 51.42, 9: 57.12, 10: 62.81, 11: 68.83,
    },
    "trend": {  # det_order = 1
        1: 3.74, 2: 16.87, 3: 23.78, 4: 30.33, 5: 36.41,
        6: 42.48, 7: 48.45, 8: 54.25, 9: 60.29, 10: 66.10, 11: 71.68,
    },
}

# 1% critical values
_OL_TRACE_1PCT: dict[str, dict[int, float]] = {
    "none": {
        1: 6.51, 2: 16.31, 3: 29.75, 4: 45.58, 5: 66.52,
        6: 90.45, 7: 119.80, 8: 152.32, 9: 187.31, 10: 226.40, 11: 269.81,
    },
    "const": {
        1: 6.65, 2: 20.04, 3: 35.65, 4: 54.46, 5: 76.07,
        6: 103.18, 7: 133.57, 8: 168.36, 9: 204.95, 10: 247.18, 11: 293.44,
    },
    "trend": {
        1: 6.40, 2: 23.46, 3: 40.49, 4: 61.21, 5: 85.78,
        6: 114.36, 7: 146.99, 8: 182.51, 9: 222.46, 10: 263.94, 11: 312.58,
    },
}

_OL_MAXEIG_1PCT: dict[str, dict[int, float]] = {
    "none": {
        1: 6.51, 2: 15.69, 3: 22.99, 4: 28.82, 5: 35.17,
        6: 41.00, 7: 47.15, 8: 53.90, 9: 59.78, 10: 65.21, 11: 72.36,
    },
    "const": {
        1: 6.65, 2: 18.63, 3: 25.52, 4: 32.24, 5: 38.77,
        6: 45.10, 7: 51.57, 8: 57.69, 9: 62.80, 10: 69.09, 11: 75.95,
    },
    "trend": {
        1: 6.40, 2: 21.47, 3: 28.83, 4: 35.68, 5: 41.58,
        6: 48.17, 7: 54.48, 8: 60.81, 9: 66.91, 10: 72.96, 11: 78.51,
    },
}


def _det_order_key(det_order: int) -> str:
    """Map statsmodels det_order (-1, 0, 1) to table key."""
    return {-1: "none", 0: "const", 1: "trend"}[det_order]


def _ol_cv(trace_or_max: str, det_order: int, n: int, signif: float = 0.05) -> float:
    """Lookup Osterwald-Lenum critical value."""
    key = _det_order_key(det_order)
    if trace_or_max == "trace":
        tbl = _OL_TRACE_1PCT if signif == 0.01 else _OL_TRACE
    else:
        tbl = _OL_MAXEIG_1PCT if signif == 0.01 else _OL_MAXEIG
    return tbl[key].get(n, float("nan"))


# ── VAR fit ────────────────────────────────────────────────────────

def var_fit(
    endog: pd.DataFrame | np.ndarray,
    *,
    lags: int | None = None,
    trend: Literal["n", "c", "ct", "ctt"] = "c",
    **fit_kwargs: Any,
) -> VARResult:
    """Estimate a VAR(p) model.

    Wraps ``statsmodels.tsa.VAR``.  Default IC convention includes all
    parameters in the penalty (matching Stata ``varsoc`` standard mode,
    R ``VARselect``, and statsmodels).  ``lutstats`` variants exclude
    deterministic terms (matching Stata ``lutstats`` option).

    Parameters
    ----------
    endog : DataFrame or ndarray
        T x K matrix of endogenous variables.
    lags : int, optional
        Number of lags.  If None, determined by info criteria.
    trend : {"n", "c", "ct", "ctt"}, default "c"
        Deterministic trend specification.
    """
    call = _capture_call(
        endog="dataframe" if isinstance(endog, pd.DataFrame) else "array",
        lags=lags, trend=trend, **fit_kwargs,
    )

    if isinstance(endog, pd.DataFrame):
        names = list(endog.columns)
        data = endog.values.astype(float)
    else:
        data = np.asarray(endog, dtype=float)
        names = [f"y{i}" for i in range(data.shape[1])]

    model = VAR(data)
    model.names = names

    if lags is None:
        sel = model.select_order(maxlags=fit_kwargs.pop("maxlags", None), trend=trend)
        lags = sel.aic

    sm_result = model.fit(lags, trend=trend, **fit_kwargs)

    # Compute lutstats IC (exclude deterministic params from penalty)
    arparms = sm_result.k_ar * sm_result.neqs ** 2
    T = sm_result.nobs
    logdet = _logdet_symm(sm_result.sigma_u_mle)

    aic_lut = logdet + (2 * arparms) / T
    bic_lut = logdet + (math.log(T) / T) * arparms
    hqic_lut = logdet + (2 * math.log(math.log(T)) / T) * arparms

    return VARResult(
        k_ar=sm_result.k_ar,
        neqs=sm_result.neqs,
        nobs=sm_result.nobs,
        n_totobs=sm_result.n_totobs,
        coefs=sm_result.coefs,
        sigma_u=sm_result.sigma_u,
        params=sm_result.params,
        llf=float(sm_result.llf),
        aic=float(sm_result.aic),
        bic=float(sm_result.bic),
        hqic=float(sm_result.hqic),
        fpe=float(sm_result.fpe),
        aic_lutstats=float(aic_lut),
        bic_lutstats=float(bic_lut),
        hqic_lutstats=float(hqic_lut),
        residuals=pd.DataFrame(sm_result.resid, columns=names),
        names=names,
        trend=trend,
        _sm_result=sm_result,
        call=call,
    )


def _logdet_symm(m: np.ndarray) -> float:
    """Log-determinant of a symmetric matrix."""
    sign, logdet = np.linalg.slogdet(m)
    return float(logdet)


def _count_deterministic_params(sm_result, trend: str) -> int:
    """Count the number of deterministic parameters in the VAR."""
    # The trend column(s) in the lagged regressor matrix
    k_trend_map = {"n": 0, "c": 1, "ct": 2, "ctt": 3}
    k_trend = k_trend_map.get(trend, 1)
    # Each equation has k_trend deterministic params, across neqs equations
    return k_trend * sm_result.neqs


# ── Lag order selection ───────────────────────────────────────────

def var_select_order(
    endog: pd.DataFrame | np.ndarray,
    *,
    maxlags: int | None = None,
    trend: Literal["n", "c", "ct", "ctt"] = "c",
) -> LagOrderResult:
    """Select VAR lag order by information criteria.

    Provides two IC conventions:
    - **Standard**: all parameters in the penalty (matching Stata ``varsoc``,
      R ``VARselect``, and statsmodels).
    - **Lutkepohl**: only AR parameters in the penalty (matching Stata's
      ``lutstats`` option, following Lutkepohl 2005).

    Parameters
    ----------
    endog : DataFrame or ndarray
        T x K matrix of endogenous variables.
    maxlags : int, optional
        Maximum number of lags to consider.  Default: ``int(round(12 * (T/100)**0.25))``.
    trend : {"n", "c", "ct", "ctt"}, default "c"
        Deterministic trend specification.
    """
    call = _capture_call(
        endog="dataframe" if isinstance(endog, pd.DataFrame) else "array",
        maxlags=maxlags, trend=trend,
    )

    if isinstance(endog, pd.DataFrame):
        names = list(endog.columns)
        data = endog.values.astype(float)
    else:
        data = np.asarray(endog, dtype=float)
        names = [f"y{i}" for i in range(data.shape[1])]

    neqs = data.shape[1]
    T_total = data.shape[0]

    model = VAR(data)
    model.names = names
    if maxlags is None:
        maxlags = int(round(12 * (T_total / 100) ** 0.25))

    sm_sel = model.select_order(maxlags=maxlags, trend=trend)

    # Standard IC values (from statsmodels)
    ic_values = {k: list(v) for k, v in sm_sel.ics.items()}
    selected = {
        "aic": int(sm_sel.aic),
        "bic": int(sm_sel.bic),
        "hqic": int(sm_sel.hqic),
        "fpe": int(sm_sel.fpe),
    }

    # Lutkepohl IC: recompute at each lag with only AR params in penalty
    ic_values_lut: dict[str, list[float]] = {"aic": [], "bic": [], "hqic": []}

    for p in range(1, maxlags + 1):
        offset = maxlags - p
        try:
            res_p = model._estimate_var(p, offset=offset, trend=trend)
        except Exception:
            ic_values_lut["aic"].append(float("nan"))
            ic_values_lut["bic"].append(float("nan"))
            ic_values_lut["hqic"].append(float("nan"))
            continue

        logdet = _logdet_symm(res_p.sigma_u_mle)
        arparms = p * neqs ** 2
        T_p = res_p.nobs

        ic_values_lut["aic"].append(logdet + (2 * arparms) / T_p)
        ic_values_lut["bic"].append(logdet + (math.log(T_p) / T_p) * arparms)
        ic_values_lut["hqic"].append(logdet + (2 * math.log(math.log(T_p)) / T_p) * arparms)

    selected_lut = {}
    for ic in ["aic", "bic", "hqic"]:
        vals = ic_values_lut[ic]
        best_idx = int(np.nanargmin(vals))
        selected_lut[ic] = best_idx + 1  # 1-indexed lag

    return LagOrderResult(
        ic_values=ic_values,
        selected=selected,
        selected_lutstats=selected_lut,
        ic_values_lutstats=ic_values_lut,
        maxlags=maxlags,
        neqs=neqs,
        nobs=T_total,
        trend=trend,
        call=call,
    )


# ── Johansen cointegration ───────────────────────────────────────

# Mapping from statsmodels det_order to the 5 Johansen cases
# det_order: -1=none, 0=const, 1=trend
# Case 1: no constant, no trend (det_order=-1, no exog)
# Case 2: restricted constant (det_order=0 in VECM sense)
# Case 3: unrestricted constant (det_order=0, VECM ci)
# Case 4: restricted trend
# Case 5: unrestricted trend (det_order=1)

def johansen_cointegration(
    endog: pd.DataFrame | np.ndarray,
    *,
    det_order: int = 0,
    k_ar_diff: int = 1,
    signif: float = 0.05,
) -> JohansenResult:
    """Johansen cointegration test with Osterwald-Lenum (1992) critical values.

    Default CV table matches Stata ``vecrank`` and R ``urca::ca.jo``.
    Statsmodels' native MacKinnon (1996) CVs are also provided for reference.

    Parameters
    ----------
    endog : DataFrame or ndarray
        T x K matrix of endogenous (levels) variables.
    det_order : {-1, 0, 1}, default 0
        Deterministic trend order: -1 = none, 0 = constant, 1 = linear trend.
    k_ar_diff : int, default 1
        Number of lagged differences in the VECM.
    signif : float, default 0.05
        Significance level for rank selection (0.05 or 0.01).
    """
    call = _capture_call(
        endog="dataframe" if isinstance(endog, pd.DataFrame) else "array",
        det_order=det_order, k_ar_diff=k_ar_diff, signif=signif,
    )

    if isinstance(endog, pd.DataFrame):
        neqs = endog.shape[1]
        data = endog.values.astype(float)
    else:
        data = np.asarray(endog, dtype=float)
        neqs = data.shape[1]

    # Run statsmodels Johansen test (for statistics and MacKinnon CVs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

    trace_stat = np.asarray(sm_result.lr1)
    max_eig_stat = np.asarray(sm_result.lr2)
    cvt_mackinnon = np.asarray(sm_result.cvt)  # (neqs x 3): 90%, 95%, 99%
    cvm_mackinnon = np.asarray(sm_result.cvm)

    # Build Osterwald-Lenum CVs (neqs x 3): 90%, 95%, 99%
    cvt_ol = np.full((neqs, 3), np.nan)
    cvm_ol = np.full((neqs, 3), np.nan)
    for i in range(neqs):
        n = neqs - i  # K-r where r=i
        if n >= 1 and n <= 11:
            cvt_ol[i, 0] = _ol_cv("trace", det_order, n, 0.10)
            cvt_ol[i, 1] = _ol_cv("trace", det_order, n, 0.05)
            cvt_ol[i, 2] = _ol_cv("trace", det_order, n, 0.01)
            cvm_ol[i, 0] = _ol_cv("maxeig", det_order, n, 0.10)
            cvm_ol[i, 1] = _ol_cv("maxeig", det_order, n, 0.05)
            cvm_ol[i, 2] = _ol_cv("maxeig", det_order, n, 0.01)

    # Determine rank via sequential trace test at chosen significance
    sig_idx = {0.10: 0, 0.05: 1, 0.01: 2}[signif]
    trace_rank = 0
    for r in range(neqs):
        if trace_stat[r] > cvt_ol[r, sig_idx]:
            trace_rank = r + 1
        else:
            break

    max_rank = 0
    for r in range(neqs):
        if max_eig_stat[r] > cvm_ol[r, sig_idx]:
            max_rank = r + 1
        else:
            break

    nobs = data.shape[0] - k_ar_diff - 1

    return JohansenResult(
        trace_stat=pd.Series(trace_stat, index=[f"r<={i}" for i in range(neqs)]),
        max_eig_stat=pd.Series(max_eig_stat, index=[f"r<={i}" for i in range(neqs)]),
        cvt=pd.DataFrame(cvt_ol, index=[f"r<={i}" for i in range(neqs)], columns=["10%", "5%", "1%"]),
        cvm=pd.DataFrame(cvm_ol, index=[f"r<={i}" for i in range(neqs)], columns=["10%", "5%", "1%"]),
        cvt_mackinnon=pd.DataFrame(cvt_mackinnon, index=[f"r<={i}" for i in range(neqs)], columns=["90%", "95%", "99%"]),
        cvm_mackinnon=pd.DataFrame(cvm_mackinnon, index=[f"r<={i}" for i in range(neqs)], columns=["90%", "95%", "99%"]),
        eigvals=sm_result.eig,
        neqs=neqs,
        k_ar_diff=k_ar_diff,
        det_order=det_order,
        nobs=nobs,
        trace_stat_rank=trace_rank,
        max_eig_stat_rank=max_rank,
        call=call,
    )


# ── Johansen rank selection via IC ───────────────────────────────

def _select_rank_ic(
    endog: np.ndarray, det_order: int, k_ar_diff: int, neqs: int,
) -> dict[str, int]:
    """Select cointegration rank by minimizing IC over ranks 0..neqs."""
    from statsmodels.tsa.vector_ar.vecm import VECM

    ic_results = {"aic": [], "bic": [], "hqic": []}

    for r in range(neqs + 1):
        if r == 0:
            # No cointegration: just collect VAR IC for comparison
            try:
                from statsmodels.tsa.vector_ar.var_model import VAR
                model = VAR(endog)
                sel = model.select_order(maxlags=k_ar_diff + 1, trend=_det_to_var_trend(det_order))
                ic_results["aic"].append(getattr(sel, "aic", float("nan")))
                ic_results["bic"].append(getattr(sel, "bic", float("nan")))
                ic_results["hqic"].append(getattr(sel, "hqic", float("nan")))
            except Exception:
                for k in ic_results:
                    ic_results[k].append(float("nan"))
        else:
            try:
                det = _det_order_to_vecm_det(det_order)
                v = VECM(endog, deterministic=det, k_ar_diff=k_ar_diff, coint_rank=r)
                res = v.fit()
                # Approximate IC from log-likelihood
                T = res.nobs
                # Count free params: alpha (neqs x r), beta (neqs x r - r*(r-1)/2 orthog),
                # gamma (neqs x neqs*(k_ar_diff)), deterministic
                n_alpha = neqs * r
                n_beta_free = r * (neqs - r)  # reduced rank
                n_gamma = neqs * neqs * k_ar_diff
                n_det = _count_vecm_det(det, neqs)
                parms = n_alpha + n_beta_free + n_gamma + n_det
                ll = res.llf
                ic_results["aic"].append((-2 * ll + 2 * parms) / T)
                ic_results["bic"].append((-2 * ll + math.log(T) * parms) / T)
                ic_results["hqic"].append((-2 * ll + 2 * math.log(math.log(T)) * parms) / T)
            except Exception:
                for k in ic_results:
                    ic_results[k].append(float("nan"))

    selected = {}
    for ic in ["aic", "bic", "hqic"]:
        vals = ic_results[ic]
        best_idx = int(np.nanargmin(vals))
        selected[ic] = best_idx

    return selected


def _det_order_to_vecm_det(det_order: int) -> str:
    """Map statsmodels det_order to VECM deterministic string."""
    return {-1: "n", 0: "co", 1: "colo"}.get(det_order, "co")


def _det_to_var_trend(det_order: int) -> str:
    """Map det_order to VAR trend string."""
    return {-1: "n", 0: "c", 1: "ct"}.get(det_order, "c")


def _count_vecm_det(det: str, neqs: int) -> int:
    """Count deterministic parameters in a VECM specification."""
    n = 0
    if "co" in det:
        n += neqs  # unrestricted constant
    if "ci" in det:
        n += 1  # restricted constant (1 per cointegrating relation)
    if "lo" in det:
        n += neqs  # unrestricted trend
    if "li" in det:
        n += 1  # restricted trend
    return n


# ── Granger causality ────────────────────────────────────────────

def granger_causality(
    var_result: VARResult,
    caused: int | str | list,
    causing: int | str | list | None = None,
    *,
    kind: Literal["f", "wald"] = "f",
    signif: float = 0.05,
) -> GrangerResult:
    """Granger causality test.

    Default is the F-test (matching R ``vars::causality`` and statsmodels),
    which has better small-sample properties.  ``kind="wald"`` gives the
    asymptotic chi-squared test (matching Stata ``vargranger`` default).

    Parameters
    ----------
    var_result : VARResult
        A fitted VAR model from ``var_fit()``.
    caused : int or str or list
        Variable(s) being tested as caused.
    causing : int or str or list or None
        Variable(s) suspected of causing.  If None, uses the complement.
    kind : {"f", "wald"}, default "f"
        Test statistic type.
    signif : float, default 0.05
        Significance level.
    """
    call = _capture_call(
        kind=kind, signif=signif,
    )

    sm_test = var_result.test_causality(caused, causing, kind=kind, signif=signif)

    method_label = "F" if kind == "f" else "Wald (chi-squared)"

    return GrangerResult(
        test_name="Granger Causality",
        test_statistic=float(sm_test.test_statistic),
        df=sm_test.df,
        pvalue=float(sm_test.pvalue),
        caused=list(sm_test.caused),
        causing=list(sm_test.causing),
        method=method_label,
        signif=signif,
        conclusion=sm_test.conclusion,
        call=call,
    )


def instantaneous_causality(
    var_result: VARResult,
    causing: int | str | list,
    *,
    signif: float = 0.05,
) -> GrangerResult:
    """Instantaneous causality test (chi-squared).

    Matches R ``vars::causality(, test="Instant")``.  Tests whether
    the residual covariance between ``causing`` and the remaining
    variables is zero.

    Parameters
    ----------
    var_result : VARResult
        A fitted VAR model from ``var_fit()``.
    causing : int or str or list
        Variable(s) suspected of causing instantaneous causality.
    signif : float, default 0.05
        Significance level.
    """
    call = _capture_call(signif=signif)

    sm_test = var_result.test_inst_causality(causing, signif=signif)

    return GrangerResult(
        test_name="Instantaneous Causality",
        test_statistic=float(sm_test.test_statistic),
        df=int(sm_test.df),
        pvalue=float(sm_test.pvalue),
        caused=list(sm_test.caused),
        causing=list(sm_test.causing),
        method="Chi-squared",
        signif=signif,
        conclusion=sm_test.conclusion,
        call=call,
    )


# ── VECM estimation ──────────────────────────────────────────────

def vecm_fit(
    endog: pd.DataFrame | np.ndarray,
    *,
    k_ar_diff: int = 1,
    coint_rank: int = 1,
    deterministic: Literal["n", "co", "ci", "lo", "li", "coli", "colo", "cili", "colo"] = "co",
    seasons: int = 0,
    first_season: int = 0,
    exog: np.ndarray | None = None,
    exog_coint: np.ndarray | None = None,
) -> VECMResult:
    """Estimate a VECM via ``statsmodels.tsa.VECM``.

    The five Johansen deterministic-trend cases map to the
    ``deterministic`` parameter as follows:

    =====  ==============  =====================================
    Case   deterministic   Description
    =====  ==============  =====================================
    I      ``"n"``         No constant, no trend
    II     ``"ci"``        Restricted constant (in coint. eq.)
    III    ``"co"``        Unrestricted constant
    IV     ``"coli"``      Unrestricted constant + restricted trend
    V      ``"colo"``      Unrestricted constant + unrestricted trend
    =====  ==============  =====================================

    Parameters
    ----------
    endog : DataFrame or ndarray
        T x K matrix of endogenous (levels) variables.
    k_ar_diff : int, default 1
        Number of lagged differences (VECM lag = k_ar - 1 in VAR terms).
    coint_rank : int, default 1
        Cointegration rank (number of cointegrating relations).
    deterministic : str, default "co"
        Deterministic term specification.
    """
    call = _capture_call(
        endog="dataframe" if isinstance(endog, pd.DataFrame) else "array",
        k_ar_diff=k_ar_diff, coint_rank=coint_rank,
        deterministic=deterministic, seasons=seasons,
    )

    if isinstance(endog, pd.DataFrame):
        names = list(endog.columns)
        data = endog.values.astype(float)
    else:
        data = np.asarray(endog, dtype=float)
        names = [f"y{i}" for i in range(data.shape[1])]

    v = VECM(
        data,
        deterministic=deterministic,
        k_ar_diff=k_ar_diff,
        coint_rank=coint_rank,
        seasons=seasons,
        first_season=first_season,
        exog=exog,
        exog_coint=exog_coint,
    )

    sm_result = v.fit()

    return VECMResult(
        alpha=sm_result.alpha,
        beta=sm_result.beta,
        gamma=sm_result.gamma,
        sigma_u=sm_result.sigma_u,
        det_coef_coint=sm_result.det_coef_coint,
        det_coef=sm_result.det_coef,
        llf=float(sm_result.llf),
        nobs=sm_result.nobs,
        neqs=sm_result.neqs,
        k_ar=sm_result.k_ar,
        coint_rank=coint_rank,
        deterministic=deterministic,
        residuals=pd.DataFrame(sm_result.resid, columns=names) if hasattr(sm_result, "resid") else None,
        _sm_result=sm_result,
        call=call,
    )


# ── vec2var: VECM to VAR conversion ──────────────────────────────

def vec2var(
    vecm_result: VECMResult,
    rank: int | None = None,
) -> VARResult:
    """Convert a VECM to a VAR in levels.

    Matches R ``vars::vec2var(z, r)``.  The conversion uses the
    VAR representation of the VECM for post-cointegration Granger
    causality testing.

    Parameters
    ----------
    vecm_result : VECMResult
        A fitted VECM from ``vecm_fit()``.
    rank : int, optional
        Cointegration rank to use.  If None, uses the rank from
        the VECM estimation.
    """
    call = _capture_call(rank=rank)

    if rank is None:
        rank = vecm_result.coint_rank

    # Get the VAR representation from the VECM results
    sm_v = vecm_result._sm_result
    var_rep_coefs = sm_v.var_rep()  # (k_ar x neqs x neqs)

    # The VECM's sigma_u is already the VAR residual covariance
    sigma_u = sm_v.sigma_u

    # Reconstruct VAR parameter vector
    # The VAR representation has: Y_t = A_1 Y_{t-1} + ... + A_p Y_{t-p} + C + e_t
    # where A_i come from var_rep() and C from det_coef
    neqs = vecm_result.neqs
    k_ar = vecm_result.k_ar

    # Build the full params array for VARResults compatibility
    # params shape: (Kp + n_determ, K) = (neqs*k_ar + n_det, neqs)
    n_det = 0
    det_str = vecm_result.deterministic
    if "co" in det_str:
        n_det += neqs
    if "lo" in det_str:
        n_det += neqs
    # Note: ci/li contribute to coint eq, not VAR intercept

    params = np.zeros((neqs * k_ar + n_det, neqs))
    # Fill AR coefficients (row-major stacking)
    for i in range(k_ar):
        params[i * neqs:(i + 1) * neqs, :] = var_rep_coefs[i].T
    # Fill deterministic (intercept/trend)
    if n_det > 0 and hasattr(sm_v, "det_coef"):
        det_vals = sm_v.det_coef
        if det_vals is not None and det_vals.size > 0:
            params[neqs * k_ar:, :] = det_vals.T

    # Build VARResults-compatible object
    # We use the underlying statsmodels object for method dispatch
    # but create a minimal wrapper

    # Get the original data for VAR construction
    endog = sm_v.endog.T if hasattr(sm_v, "endog") else None  # VECM stores transposed

    from statsmodels.tsa.vector_ar.var_model import VARResults

    # Use the VECM's data to fit a VAR at the appropriate lag
    # This is the cleanest way to get a proper VARResults object
    if endog is not None:
        # Re-fit as VAR using the VECM's lag structure
        model = VAR(endog)
        # We need to create a VARResults with the VECM-derived parameters
        # The simplest approach: fit a VAR at k_ar lags with the VECM's sigma_u
        sm_var = VARResults(
            endog=endog,
            endog_lagged=None,  # Will be computed
            params=params,
            sigma_u=sigma_u,
            lag_order=k_ar,
            model=model,
            trend=vecm_result.deterministic,
            names=vecm_result._sm_result.model.names if hasattr(vecm_result._sm_result, "model") else None,
        )
    else:
        # Fallback: create a minimal object
        sm_var = _MinimalVARResult(
            params=params, sigma_u=sigma_u, k_ar=k_ar, neqs=neqs,
            coefs=var_rep_coefs, llf=vecm_result.llf,
        )

    names = vecm_result._sm_result.model.names if hasattr(vecm_result._sm_result, "model") else [f"y{i}" for i in range(neqs)]

    return VARResult(
        k_ar=k_ar,
        neqs=neqs,
        nobs=vecm_result.nobs,
        n_totobs=vecm_result.nobs + k_ar,
        coefs=var_rep_coefs,
        sigma_u=sigma_u,
        params=params,
        llf=vecm_result.llf,
        aic=float("nan"),  # VAR IC not directly meaningful for VECM-derived VAR
        bic=float("nan"),
        hqic=float("nan"),
        fpe=float("nan"),
        aic_lutstats=float("nan"),
        bic_lutstats=float("nan"),
        hqic_lutstats=float("nan"),
        residuals=pd.DataFrame(np.zeros((1, neqs)), columns=names),  # placeholder
        names=names,
        trend=vecm_result.deterministic,
        _sm_result=sm_var,
        call=call,
    )


class _MinimalVARResult:
    """Minimal VARResults-compatible object for vec2var output."""

    def __init__(self, params, sigma_u, k_ar, neqs, coefs, llf):
        self.params = params
        self.sigma_u = sigma_u
        self.sigma_u_mle = sigma_u
        self.k_ar = k_ar
        self.neqs = neqs
        self.coefs = coefs
        self.llf = llf
        self.nobs = 0
        self.n_totobs = 0
        self.names = [f"y{i}" for i in range(neqs)]

    def test_causality(self, caused, causing=None, kind="f", signif=0.05):
        raise NotImplementedError(
            "Granger causality on vec2var output requires a full VAR refit. "
            "Use var_fit() with the VAR representation instead."
        )

    def test_inst_causality(self, causing, signif=0.05):
        raise NotImplementedError(
            "Instantaneous causality on vec2var output requires a full VAR refit."
        )

    def irf(self, periods=10, **kwargs):
        raise NotImplementedError("IRF not available on vec2var output without full VAR refit.")

    def fevd(self, periods=10, **kwargs):
        raise NotImplementedError("FEVD not available on vec2var output without full VAR refit.")

    def summary(self):
        return "VAR representation of VECM (partial results -- use var_fit for full VAR)"
