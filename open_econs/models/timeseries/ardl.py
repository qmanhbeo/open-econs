"""ARDL / UECM error-correction models + PSS (2001) bounds test.

Thin wrappers around ``statsmodels.tsa.ardl`` reconciled to Stata SSC ``ardl``
(Kripfganz & Schneider 2018) and R ``ARDL`` / ``dynamac``:

- ``ardl_fit()``      -- estimate an ARDL(p, q1, ..., qk) model.
- ``uecm_fit()``      -- unrestricted error-correction reparameterization,
                          exposing the long-run coefficients and the
                          speed-of-adjustment (EC) term.
- ``ardl_select_order()`` -- IC-based order selection.
- ``ARDLResult.bounds_test`` / ``UECMResult.bounds_test`` -- the Pesaran-Shin-
                          Smith (2001) bounds test for a level relationship,
                          returning both the F-bounds and the t-bounds.

Conventions matched (see ``methodology/timeseries/ardl.md`` for the full
source-verified crosswalk):

- **PSS cases 1-5** use the standard PSS Table CI numbering, identical across
  Stata, R and statsmodels.
- **Critical-value vintage** is toggled by ``cv_vintage``; the default
  ``"pss2001"`` (asymptotic) is the only vintage common to all four reference
  tools and the only one asserted at 1e-6 cross-tool.
- **Long-run coefficient sign** follows Stata ``ardl, ec`` / R
  ``multipliers()`` (``LR = -theta / rho``) by default; statsmodels' raw
  ``ci_params`` sign is available via ``lr_sign="statsmodels"``.
- The **t-bounds test** (on the ``y_{t-1}`` coefficient) is computed by OE
  because statsmodels' ``bounds_test`` returns only the F-type statistic.
"""
from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from open_econs.core.call_capture import capture_call as _capture_call
from open_econs.models.timeseries.results import (
    ARDLResult,
    BoundsTestResult,
    OrderSelectionResult,
    UECMResult,
)

from statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order as _sm_select_order


# PSS (2001) case -> statsmodels trend string.
_CASE_TO_TREND: dict[int, str] = {1: "n", 2: "c", 3: "c", 4: "ct", 5: "ct"}

# t-bounds case folding (Stata ardlbounds.ado L30): the t-statistic is
# unaffected by restrictions on the deterministic components, so restricted
# cases collapse onto their unrestricted sibling for the t-bounds only.
_T_CASE_FOLD: dict[int, int] = {1: 1, 2: 3, 3: 3, 4: 5, 5: 5}


def _resolve_exog(
    data: pd.DataFrame,
    exog: str | Sequence[str] | None,
) -> tuple[pd.DataFrame | None, list[str]]:
    if exog is None:
        return None, []
    names = [exog] if isinstance(exog, str) else list(exog)
    return data[names].astype(float), names


def ardl_fit(
    data: pd.DataFrame,
    y: str,
    *,
    exog: str | Sequence[str] | None = None,
    order: int | tuple[int, ...] | dict[str, int] = 0,
    lags: int | Sequence[int] = 1,
    trend: Literal["n", "c", "ct", "ctt"] = "c",
    causal: bool = False,
    **fit_kwargs: Any,
) -> ARDLResult:
    """Estimate an ARDL(p, q1, ..., qk) model.

    Wraps ``statsmodels.tsa.ardl.ARDL``.

    Parameters
    ----------
    data : DataFrame
        Working dataset (assumed already in time order).
    y : str
        Dependent-series column name.
    exog : str or sequence of str, optional
        Regressor column name(s).
    order : int or tuple or dict, default 0
        Distributed-lag order(s) of the exogenous regressors.
    lags : int or sequence of int, default 1
        Autoregressive lag order ``p`` of the dependent series.
    trend : {"n", "c", "ct", "ctt"}, default "c"
        Deterministic terms.
    causal : bool, default False
        If True, drop the contemporaneous regressor (lag 0).
    """
    call = _capture_call(
        y=y, exog=exog, order=order, lags=lags, trend=trend, causal=causal,
        **fit_kwargs,
    )
    endog = data[y].astype(float)
    exog_df, exog_names = _resolve_exog(data, exog)

    model = ARDL(
        endog, lags=lags, exog=exog_df, order=order, trend=trend, causal=causal,
    )
    res = model.fit(**fit_kwargs)

    return _build_ardl_result(res, y, exog_names, trend, order, lags, call)


def _build_ardl_result(
    res: Any,
    y: str,
    exog_names: list[str],
    trend: str,
    order: Any,
    lags: Any,
    call: dict,
) -> ARDLResult:
    params = res.params
    return ARDLResult(
        params=params,
        std_errors=res.bse.reindex(params.index),
        t_stats=res.tvalues.reindex(params.index),
        p_values=res.pvalues.reindex(params.index),
        conf_int=res.conf_int(),
        llf=float(res.llf),
        aic=float(res.aic),
        bic=float(res.bic),
        hqic=float(res.hqic),
        nobs=int(res.nobs),
        resid=pd.Series(np.asarray(res.resid), name="resid"),
        fitted_values=pd.Series(np.asarray(res.fittedvalues), name="fitted"),
        y_name=y,
        exog_names=exog_names,
        trend=trend,
        order=order,
        lags=lags,
        _sm_result=res,
        _is_uecm=False,
        call=call,
    )


def uecm_fit(
    data: pd.DataFrame,
    y: str,
    *,
    exog: str | Sequence[str] | None = None,
    order: int | tuple[int, ...] | dict[str, int] = 0,
    lags: int = 1,
    trend: Literal["n", "c", "ct", "ctt"] = "c",
    causal: bool = False,
    lr_sign: Literal["stata", "statsmodels"] = "stata",
    **fit_kwargs: Any,
) -> UECMResult:
    """Estimate the unrestricted error-correction (UECM) form of an ARDL.

    Wraps ``statsmodels.tsa.ardl.UECM``.  Exposes the long-run coefficients
    and the speed-of-adjustment (error-correction) term.

    Parameters
    ----------
    data, y, exog, order, lags, trend, causal
        As in :func:`ardl_fit` (``lags`` must be a scalar for the UECM).
    lr_sign : {"stata", "statsmodels"}, default "stata"
        Sign convention for the long-run coefficients.  ``"stata"`` reports
        ``-theta / rho`` (matching Stata ``ardl, ec`` and R
        ``ARDL::multipliers()``); ``"statsmodels"`` returns the raw
        ``ci_params`` (opposite sign, ``y.L1`` normalized to 1.0).
    """
    call = _capture_call(
        y=y, exog=exog, order=order, lags=lags, trend=trend, causal=causal,
        lr_sign=lr_sign, **fit_kwargs,
    )
    endog = data[y].astype(float)
    exog_df, exog_names = _resolve_exog(data, exog)

    model = UECM(
        endog, lags=lags, exog=exog_df, order=order, trend=trend, causal=causal,
    )
    res = model.fit(**fit_kwargs)

    params = res.params

    # Long-run coefficients.  statsmodels ci_params reports +theta/rho with the
    # y.L1 entry normalized to 1.0; Stata/R report -theta/rho excluding that
    # base.  See methodology/timeseries/ardl.md section 5.
    ci = res.ci_params
    long_run = _long_run_coefficients(ci, lr_sign)

    # Speed-of-adjustment term = raw coefficient on the level y_{t-1}.
    ec_name = _ec_term_name(params.index)
    ec_term = float(params[ec_name]) if ec_name is not None else float("nan")
    ec_se = float(res.bse[ec_name]) if ec_name is not None else float("nan")
    ec_t = float(res.tvalues[ec_name]) if ec_name is not None else float("nan")
    ec_p = float(res.pvalues[ec_name]) if ec_name is not None else float("nan")

    return UECMResult(
        params=params,
        std_errors=res.bse.reindex(params.index),
        t_stats=res.tvalues.reindex(params.index),
        p_values=res.pvalues.reindex(params.index),
        conf_int=res.conf_int(),
        llf=float(res.llf),
        aic=float(res.aic),
        bic=float(res.bic),
        hqic=float(res.hqic),
        nobs=int(res.nobs),
        resid=pd.Series(np.asarray(res.resid), name="resid"),
        fitted_values=pd.Series(np.asarray(res.fittedvalues), name="fitted"),
        y_name=y,
        exog_names=exog_names,
        trend=trend,
        order=order,
        lags=lags,
        long_run=long_run,
        ec_term=ec_term,
        ec_term_se=ec_se,
        ec_term_t=ec_t,
        ec_term_pvalue=ec_p,
        ec_term_name=ec_name if ec_name is not None else "",
        lr_sign=lr_sign,
        _sm_result=res,
        call=call,
    )


def _ec_term_name(index: pd.Index) -> str | None:
    """Locate the level y_{t-1} term (the speed-of-adjustment coefficient)."""
    for name in index:
        s = str(name)
        if s.endswith(".L1") and not s.startswith("D."):
            # First non-differenced .L1 term is the level y_{t-1}.
            # statsmodels orders levels as [y.L1, x1.L1, ...]; y is first.
            return str(name)
    return None


def _long_run_coefficients(
    ci_params: pd.Series,
    lr_sign: str,
) -> pd.Series:
    """Build the reported long-run coefficient vector from ci_params.

    statsmodels ci_params has the dependent variable normalized to 1.0 and
    reports +theta/rho for the regressors.  Stata/R report -theta/rho and omit
    the normalized 1.0.  We drop the dependent (==1.0) entry and apply the sign.
    """
    ci = ci_params.copy()
    # The dependent variable's normalized entry equals 1.0 exactly.
    depvar_mask = np.isclose(ci.values, 1.0)
    # Only drop the single normalization base (first exact 1.0), keep the rest.
    keep = np.ones(len(ci), dtype=bool)
    if depvar_mask.any():
        keep[np.argmax(depvar_mask)] = False
    lr = ci[keep]
    if lr_sign == "stata":
        lr = -lr
    return pd.Series(lr.values, index=lr.index, name="long_run")


def ardl_select_order(
    data: pd.DataFrame,
    y: str,
    *,
    exog: str | Sequence[str] | None = None,
    maxlag: int = 4,
    maxorder: int | dict[str, int] = 4,
    trend: Literal["n", "c", "ct", "ctt"] = "c",
    ic: Literal["aic", "bic"] = "bic",
    glob: bool = False,
    causal: bool = False,
) -> OrderSelectionResult:
    """Select ARDL order by information criterion.

    Wraps ``statsmodels.tsa.ardl.ardl_select_order``.  Default ``ic="bic"``
    matches Stata ``ardl`` and statsmodels; R ``ARDL::auto_ardl`` defaults to
    AIC, so pin ``ic=`` explicitly for cross-tool parity.

    Parameters
    ----------
    maxlag : int, default 4
        Maximum autoregressive lag (Stata ``ardl`` default is 4).
    maxorder : int or dict, default 4
        Maximum distributed-lag order.
    ic : {"aic", "bic"}, default "bic"
        Information criterion to minimize.
    glob : bool, default False
        If True, search over all lag combinations rather than only the
        symmetric grid.
    """
    call = _capture_call(
        y=y, exog=exog, maxlag=maxlag, maxorder=maxorder, trend=trend,
        ic=ic, glob=glob, causal=causal,
    )
    endog = data[y].astype(float)
    exog_df, exog_names = _resolve_exog(data, exog)

    sel = _sm_select_order(
        endog, maxlag=maxlag, exog=exog_df, maxorder=maxorder, trend=trend,
        ic=ic, glob=glob, causal=causal,
    )

    best = sel.model
    ar_order = int(best.ar_lags[-1]) if best.ar_lags else 0
    dl_orders = {k: (max(v) if v else 0) for k, v in (best.dl_lags or {}).items()}

    # sel.<ic> is a Series over all candidate models; the selected model's IC
    # value is read off the fitted best model itself.
    ic_value = float(getattr(best.fit(), ic))

    return OrderSelectionResult(
        selected_ar_order=ar_order,
        selected_dl_orders=dl_orders,
        ic=ic,
        ic_value=ic_value,
        maxlag=maxlag,
        maxorder=maxorder,
        trend=trend,
        y_name=y,
        exog_names=exog_names,
        _sm_selection=sel,
        call=call,
    )


# ── PSS bounds test ──────────────────────────────────────────────

def bounds_test(
    result: ARDLResult | UECMResult,
    case: int,
    *,
    cv_vintage: str = "pss2001",
    signif: Sequence[float] = (0.10, 0.05, 0.01),
) -> BoundsTestResult:
    """Pesaran-Shin-Smith (2001) bounds test for a level relationship.

    Returns both the F-bounds and the t-bounds test.  The F-statistic is the
    one cross-tool agreement point (asserted at 1e-6 vs Stata / R / statsmodels).
    The t-bounds is computed by OE on the ``y_{t-1}`` coefficient because
    statsmodels' ``bounds_test`` returns only the F-type statistic; for the
    t-bounds, restricted cases fold onto their unrestricted sibling
    (2->3, 4->5), matching Stata.

    Parameters
    ----------
    result : ARDLResult or UECMResult
        A fitted ARDL/UECM model.
    case : {1, 2, 3, 4, 5}
        PSS deterministic case (standard PSS Table CI numbering).
    cv_vintage : {"pss2001", "narayan2005"}, default "pss2001"
        Critical-value table.  ``"pss2001"`` is the asymptotic PSS table (the
        cross-tool parity anchor).  ``"narayan2005"`` selects finite-sample
        critical values (documented divergence, not a cross-tool anchor).
    signif : sequence of float, default (0.10, 0.05, 0.01)
        Significance levels to report bounds for.
    """
    if case not in (1, 2, 3, 4, 5):
        raise ValueError(f"case must be 1-5, got {case}")

    call = _capture_call(case=case, cv_vintage=cv_vintage, signif=tuple(signif))

    # statsmodels bounds_test lives on UECMResults; convert if given an ARDL.
    sm_res = result._sm_result
    if result.__class__.__name__ == "ARDLResult":
        from statsmodels.tsa.ardl import UECM
        sm_res = UECM.from_ardl(sm_res.model).fit()

    asymptotic = cv_vintage == "pss2001"
    bt = sm_res.bounds_test(case=case, asymptotic=asymptotic)

    f_stat = float(bt.stat)
    cv = bt.crit_vals  # DataFrame indexed by percentile, cols lower/upper
    f_lower, f_upper = _extract_bounds(cv, signif)

    # ── t-bounds (OE-computed): t on the level y_{t-1} coefficient ──
    k = len(result.exog_names)
    t_case = _T_CASE_FOLD[case]
    ec_name = _ec_term_name(sm_res.params.index)
    if ec_name is not None:
        t_stat: float | None = float(sm_res.tvalues[ec_name])
        t_lower, t_upper = _t_bounds_cv(k, t_case, signif)
    else:
        t_stat, t_lower, t_upper = None, {}, {}

    return BoundsTestResult(
        case=case,
        cv_vintage=cv_vintage,
        f_stat=f_stat,
        f_crit_lower=f_lower,
        f_crit_upper=f_upper,
        f_pvalues={
            "lower": float(bt.p_values["lower"]),
            "upper": float(bt.p_values["upper"]),
        },
        t_stat=t_stat,
        t_crit_lower=t_lower,
        t_crit_upper=t_upper,
        t_case=t_case,
        k=k,
        nobs=int(result.nobs),
        null=str(bt.null),
        alternative=str(bt.alternative),
        call=call,
    )


def _extract_bounds(
    cv: pd.DataFrame,
    signif: Sequence[float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Map significance levels to lower/upper F-bounds from a statsmodels CV DF."""
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for s in signif:
        pct = round((1.0 - s) * 100.0, 4)
        key = f"{int(round(s * 100))}%"
        if pct in cv.index:
            lower[key] = float(cv.loc[pct, "lower"])
            upper[key] = float(cv.loc[pct, "upper"])
        else:
            nearest = cv.index[np.argmin(np.abs(np.asarray(cv.index) - pct))]
            lower[key] = float(cv.loc[nearest, "lower"])
            upper[key] = float(cv.loc[nearest, "upper"])
    return lower, upper


# ── PSS (2001) t-bounds critical values (Table CII), cases I/III/V ──
# Source: Pesaran, Shin & Smith (2001), Table CII(i)/(iii)/(v).
# Keyed by k (number of regressors) -> {signif%: (I(0), I(1))}.
# t-bounds folds restricted cases: case 2->3, 4->5 (see _T_CASE_FOLD).
_PSS_T_BOUNDS: dict[int, dict[int, dict[str, tuple[float, float]]]] = {
    # case 1 (no intercept, no trend) -- Table CII(i)
    1: {
        0: {"10%": (-1.62, -1.62), "5%": (-1.95, -1.95), "1%": (-2.58, -2.58)},
        1: {"10%": (-1.62, -2.28), "5%": (-1.95, -2.60), "1%": (-2.58, -3.22)},
        2: {"10%": (-1.62, -2.68), "5%": (-1.95, -3.02), "1%": (-2.58, -3.66)},
        3: {"10%": (-1.62, -3.00), "5%": (-1.95, -3.34), "1%": (-2.58, -3.97)},
        4: {"10%": (-1.62, -3.26), "5%": (-1.95, -3.60), "1%": (-2.58, -4.23)},
    },
    # case 3 (unrestricted intercept, no trend) -- Table CII(iii)
    3: {
        0: {"10%": (-2.57, -2.57), "5%": (-2.86, -2.86), "1%": (-3.43, -3.43)},
        1: {"10%": (-2.57, -2.91), "5%": (-2.86, -3.22), "1%": (-3.43, -3.82)},
        2: {"10%": (-2.57, -3.21), "5%": (-2.86, -3.53), "1%": (-3.43, -4.10)},
        3: {"10%": (-2.57, -3.46), "5%": (-2.86, -3.78), "1%": (-3.43, -4.37)},
        4: {"10%": (-2.57, -3.66), "5%": (-2.86, -3.99), "1%": (-3.43, -4.60)},
    },
    # case 5 (unrestricted intercept, unrestricted trend) -- Table CII(v)
    5: {
        0: {"10%": (-3.13, -3.13), "5%": (-3.41, -3.41), "1%": (-3.96, -3.97)},
        1: {"10%": (-3.13, -3.40), "5%": (-3.41, -3.69), "1%": (-3.96, -4.26)},
        2: {"10%": (-3.13, -3.63), "5%": (-3.41, -3.95), "1%": (-3.96, -4.53)},
        3: {"10%": (-3.13, -3.84), "5%": (-3.41, -4.16), "1%": (-3.96, -4.73)},
        4: {"10%": (-3.13, -4.04), "5%": (-3.41, -4.36), "1%": (-3.96, -4.96)},
    },
}


def _t_bounds_cv(
    k: int,
    t_case: int,
    signif: Sequence[float],
) -> tuple[dict[str, float], dict[str, float]]:
    """PSS (2001) t-bounds critical values for cases 1/3/5."""
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    table = _PSS_T_BOUNDS.get(t_case, {})
    row = table.get(k)
    if row is None:
        return lower, upper
    for s in signif:
        key = f"{int(round(s * 100))}%"
        if key in row:
            lo, hi = row[key]
            lower[key] = lo
            upper[key] = hi
    return lower, upper
