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
    cv_vintage : {"pss2001", "statsmodels"}, default "pss2001"
        Critical-value source.  ``"pss2001"`` uses the **published asymptotic
        PSS (2001) Table CI / CII** critical values (extracted from R
        ``ARDL:::crit_val_bounds_pss2001``) -- this is the cross-tool 1e-6
        parity anchor and matches Stata ``ardl`` / R ``ARDL`` /
        ``dynamac::pssbounds``.  ``"statsmodels"`` returns statsmodels'
        Monte-Carlo *simulated* finite-sample F-bounds (``asymptotic=False``);
        these are NOT the published table and are a documented divergence, not
        a cross-tool anchor.
    signif : sequence of float, default (0.10, 0.05, 0.01)
        Significance levels to report bounds for.  The published PSS tables
        support 0.10, 0.05, 0.025 and 0.01.
    """
    if case not in (1, 2, 3, 4, 5):
        raise ValueError(f"case must be 1-5, got {case}")
    if cv_vintage not in ("pss2001", "statsmodels"):
        raise ValueError(
            f"cv_vintage must be 'pss2001' or 'statsmodels', got {cv_vintage!r}"
        )

    call = _capture_call(case=case, cv_vintage=cv_vintage, signif=tuple(signif))

    # statsmodels bounds_test lives on UECMResults; convert if given an ARDL.
    sm_res = result._sm_result
    if result.__class__.__name__ == "ARDLResult":
        from statsmodels.tsa.ardl import UECM
        sm_res = UECM.from_ardl(sm_res.model).fit()

    # The F-statistic itself is convention-free and identical across all four
    # reference tools (asymptotic flag only affects statsmodels' CV table).
    bt = sm_res.bounds_test(case=case, asymptotic=True)
    f_stat = float(bt.stat)

    k = len(result.exog_names)
    if cv_vintage == "pss2001":
        # Published PSS Table CI -- the cross-tool anchor.
        f_lower, f_upper = _f_bounds_cv(k, case, signif)
    else:
        # statsmodels' simulated finite-sample F-bounds (documented divergence).
        sim = sm_res.bounds_test(case=case, asymptotic=False)
        f_lower, f_upper = _extract_bounds(sim.crit_vals, signif)

    # ── t-bounds (OE-computed): t on the level y_{t-1} coefficient ──
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


def _signif_key(s: float) -> str:
    """Map a significance level to the table key ('10%'/'5%'/'2.5%'/'1%')."""
    pct = s * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"{int(round(pct))}%"
    return f"{pct:g}%"


def _table_bounds(
    table: dict[int, dict[int, dict[str, tuple[float, float]]]],
    case: int,
    k: int,
    signif: Sequence[float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Published PSS (2001) critical values for the given case/k from a table."""
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    row = table.get(case, {}).get(k)
    if row is None:
        return lower, upper
    for s in signif:
        key = _signif_key(s)
        if key in row:
            lo, hi = row[key]
            lower[key] = lo
            upper[key] = hi
    return lower, upper


def _t_bounds_cv(
    k: int,
    t_case: int,
    signif: Sequence[float],
) -> tuple[dict[str, float], dict[str, float]]:
    """PSS (2001) t-bounds critical values for cases 1/3/5."""
    return _table_bounds(_PSS_T_BOUNDS, t_case, k, signif)


def _f_bounds_cv(
    k: int,
    case: int,
    signif: Sequence[float],
) -> tuple[dict[str, float], dict[str, float]]:
    """PSS (2001) F-bounds critical values (Table CI), cases 1-5."""
    return _table_bounds(_PSS_F_BOUNDS, case, k, signif)


# ── PSS (2001) critical values (Table CI = F-bounds, Table CII = t-bounds) ──
# Source: extracted verbatim from R ``ARDL:::crit_val_bounds_pss2001`` (which
# reproduces Pesaran, Shin & Smith 2001, Tables CI(i-v) and CII(i,iii,v)),
# keyed by PSS case (1-5) -> k (number of regressors, 0-10) ->
# {signif%: (I(0) lower, I(1) upper)}.  Regenerated, never hand-typed, to avoid
# transcription error (rule 1); see methodology/timeseries/ardl.md.  These are
# the asymptotic ("pss2001") critical values -- the cross-tool 1e-6 anchor.
# The t-bounds fold restricted cases: case 2->3, 4->5 (see _T_CASE_FOLD).
_PSS_F_BOUNDS: dict[int, dict[int, dict[str, tuple[float, float]]]] = {
    1: {
        0: {"10%": (3.0, 3.0), "5%": (4.2, 4.2), "2.5%": (5.47, 5.47), "1%": (7.17, 7.17)},
        1: {"10%": (2.44, 3.28), "5%": (3.15, 4.11), "2.5%": (3.88, 4.92), "1%": (4.81, 6.02)},
        2: {"10%": (2.17, 3.19), "5%": (2.72, 3.83), "2.5%": (3.22, 4.5), "1%": (3.88, 5.3)},
        3: {"10%": (2.01, 3.1), "5%": (2.45, 3.63), "2.5%": (2.87, 4.16), "1%": (3.42, 4.84)},
        4: {"10%": (1.9, 3.01), "5%": (2.26, 3.48), "2.5%": (2.62, 3.9), "1%": (3.07, 4.44)},
        5: {"10%": (1.81, 2.93), "5%": (2.14, 3.34), "2.5%": (2.44, 3.71), "1%": (2.82, 4.21)},
        6: {"10%": (1.75, 2.87), "5%": (2.04, 3.24), "2.5%": (2.32, 3.59), "1%": (2.66, 4.05)},
        7: {"10%": (1.7, 2.83), "5%": (1.97, 3.18), "2.5%": (2.22, 3.49), "1%": (2.54, 3.91)},
        8: {"10%": (1.66, 2.79), "5%": (1.91, 3.11), "2.5%": (2.15, 3.4), "1%": (2.45, 3.79)},
        9: {"10%": (1.63, 2.75), "5%": (1.86, 3.05), "2.5%": (2.08, 3.33), "1%": (2.34, 3.68)},
        10: {"10%": (1.6, 2.72), "5%": (1.82, 2.99), "2.5%": (2.02, 3.27), "1%": (2.26, 3.6)},
    },
    2: {
        0: {"10%": (3.8, 3.8), "5%": (4.6, 4.6), "2.5%": (5.39, 5.39), "1%": (6.44, 6.44)},
        1: {"10%": (3.02, 3.51), "5%": (3.62, 4.16), "2.5%": (4.18, 4.79), "1%": (4.94, 5.58)},
        2: {"10%": (2.63, 3.35), "5%": (3.1, 3.87), "2.5%": (3.55, 4.38), "1%": (4.13, 5.0)},
        3: {"10%": (2.37, 3.2), "5%": (2.79, 3.67), "2.5%": (3.15, 4.08), "1%": (3.65, 4.66)},
        4: {"10%": (2.2, 3.09), "5%": (2.56, 3.49), "2.5%": (2.88, 3.87), "1%": (3.29, 4.37)},
        5: {"10%": (2.08, 3.0), "5%": (2.39, 3.38), "2.5%": (2.7, 3.73), "1%": (3.06, 4.15)},
        6: {"10%": (1.99, 2.94), "5%": (2.27, 3.28), "2.5%": (2.55, 3.61), "1%": (2.88, 3.99)},
        7: {"10%": (1.92, 2.89), "5%": (2.17, 3.21), "2.5%": (2.43, 3.51), "1%": (2.73, 3.9)},
        8: {"10%": (1.85, 2.85), "5%": (2.11, 3.15), "2.5%": (2.33, 3.42), "1%": (2.62, 3.77)},
        9: {"10%": (1.8, 2.8), "5%": (2.04, 3.08), "2.5%": (2.24, 3.35), "1%": (2.5, 3.68)},
        10: {"10%": (1.76, 2.77), "5%": (1.98, 3.04), "2.5%": (2.18, 3.28), "1%": (2.41, 3.61)},
    },
    3: {
        0: {"10%": (6.58, 6.58), "5%": (8.21, 8.21), "2.5%": (9.8, 9.8), "1%": (11.79, 11.79)},
        1: {"10%": (4.04, 4.78), "5%": (4.94, 5.73), "2.5%": (5.77, 6.68), "1%": (6.84, 7.84)},
        2: {"10%": (3.17, 4.14), "5%": (3.79, 4.85), "2.5%": (4.41, 5.52), "1%": (5.15, 6.36)},
        3: {"10%": (2.72, 3.77), "5%": (3.23, 4.35), "2.5%": (3.69, 4.89), "1%": (4.29, 5.61)},
        4: {"10%": (2.45, 3.52), "5%": (2.86, 4.01), "2.5%": (3.25, 4.49), "1%": (3.74, 5.06)},
        5: {"10%": (2.26, 3.35), "5%": (2.62, 3.79), "2.5%": (2.96, 4.18), "1%": (3.41, 4.68)},
        6: {"10%": (2.12, 3.23), "5%": (2.45, 3.61), "2.5%": (2.75, 3.99), "1%": (3.15, 4.43)},
        7: {"10%": (2.03, 3.13), "5%": (2.32, 3.5), "2.5%": (2.6, 3.84), "1%": (2.96, 4.26)},
        8: {"10%": (1.95, 3.06), "5%": (2.22, 3.39), "2.5%": (2.48, 3.7), "1%": (2.79, 4.1)},
        9: {"10%": (1.88, 2.99), "5%": (2.14, 3.3), "2.5%": (2.37, 3.6), "1%": (2.65, 3.97)},
        10: {"10%": (1.83, 2.94), "5%": (2.06, 3.24), "2.5%": (2.28, 3.5), "1%": (2.54, 3.86)},
    },
    4: {
        0: {"10%": (5.37, 5.37), "5%": (6.29, 6.29), "2.5%": (7.14, 7.14), "1%": (8.26, 8.26)},
        1: {"10%": (4.05, 4.49), "5%": (4.68, 5.15), "2.5%": (5.3, 5.83), "1%": (6.1, 6.73)},
        2: {"10%": (3.38, 4.02), "5%": (3.88, 4.61), "2.5%": (4.37, 5.16), "1%": (4.99, 5.85)},
        3: {"10%": (2.97, 3.74), "5%": (3.38, 4.23), "2.5%": (3.8, 4.68), "1%": (4.3, 5.23)},
        4: {"10%": (2.68, 3.53), "5%": (3.05, 3.97), "2.5%": (3.4, 4.36), "1%": (3.81, 4.92)},
        5: {"10%": (2.49, 3.38), "5%": (2.81, 3.76), "2.5%": (3.11, 4.13), "1%": (3.5, 4.63)},
        6: {"10%": (2.33, 3.25), "5%": (2.63, 3.62), "2.5%": (2.9, 3.94), "1%": (3.27, 4.39)},
        7: {"10%": (2.22, 3.17), "5%": (2.5, 3.5), "2.5%": (2.76, 3.81), "1%": (3.07, 4.23)},
        8: {"10%": (2.13, 3.09), "5%": (2.38, 3.41), "2.5%": (2.62, 3.7), "1%": (2.93, 4.06)},
        9: {"10%": (2.05, 3.02), "5%": (2.3, 3.33), "2.5%": (2.52, 3.6), "1%": (2.79, 3.93)},
        10: {"10%": (1.98, 2.97), "5%": (2.21, 3.25), "2.5%": (2.42, 3.52), "1%": (2.68, 3.84)},
    },
    5: {
        0: {"10%": (9.81, 9.81), "5%": (11.64, 11.64), "2.5%": (13.36, 13.36), "1%": (15.73, 15.73)},
        1: {"10%": (5.59, 6.26), "5%": (6.56, 7.3), "2.5%": (7.46, 8.27), "1%": (8.74, 9.63)},
        2: {"10%": (4.19, 5.06), "5%": (4.87, 5.85), "2.5%": (5.49, 6.59), "1%": (6.34, 7.52)},
        3: {"10%": (3.47, 4.45), "5%": (4.01, 5.07), "2.5%": (4.52, 5.62), "1%": (5.17, 6.36)},
        4: {"10%": (3.03, 4.06), "5%": (3.47, 4.57), "2.5%": (3.89, 5.07), "1%": (4.4, 5.72)},
        5: {"10%": (2.75, 3.79), "5%": (3.12, 4.25), "2.5%": (3.47, 4.67), "1%": (3.93, 5.23)},
        6: {"10%": (2.53, 3.59), "5%": (2.87, 4.0), "2.5%": (3.19, 4.38), "1%": (3.6, 4.9)},
        7: {"10%": (2.38, 3.45), "5%": (2.69, 3.83), "2.5%": (2.98, 4.16), "1%": (3.34, 4.63)},
        8: {"10%": (2.26, 3.34), "5%": (2.55, 3.68), "2.5%": (2.82, 4.02), "1%": (3.15, 4.43)},
        9: {"10%": (2.16, 3.24), "5%": (2.43, 3.56), "2.5%": (2.67, 3.87), "1%": (2.97, 4.24)},
        10: {"10%": (2.07, 3.16), "5%": (2.33, 3.46), "2.5%": (2.56, 3.76), "1%": (2.84, 4.1)},
    },
}


_PSS_T_BOUNDS: dict[int, dict[int, dict[str, tuple[float, float]]]] = {
    1: {
        0: {"10%": (-1.62, -1.62), "5%": (-1.95, -1.95), "2.5%": (-2.24, -2.24), "1%": (-2.58, -2.58)},
        1: {"10%": (-1.62, -2.28), "5%": (-1.95, -2.6), "2.5%": (-2.24, -2.9), "1%": (-2.58, -3.22)},
        2: {"10%": (-1.62, -2.68), "5%": (-1.95, -3.02), "2.5%": (-2.24, -3.31), "1%": (-2.58, -3.66)},
        3: {"10%": (-1.62, -3.0), "5%": (-1.95, -3.33), "2.5%": (-2.24, -3.64), "1%": (-2.58, -3.97)},
        4: {"10%": (-1.62, -3.26), "5%": (-1.95, -3.6), "2.5%": (-2.24, -3.89), "1%": (-2.58, -4.23)},
        5: {"10%": (-1.62, -3.49), "5%": (-1.95, -3.83), "2.5%": (-2.24, -4.12), "1%": (-2.58, -4.44)},
        6: {"10%": (-1.62, -3.7), "5%": (-1.95, -4.04), "2.5%": (-2.24, -4.34), "1%": (-2.58, -4.67)},
        7: {"10%": (-1.62, -3.9), "5%": (-1.95, -4.23), "2.5%": (-2.24, -4.54), "1%": (-2.58, -4.88)},
        8: {"10%": (-1.62, -4.09), "5%": (-1.95, -4.43), "2.5%": (-2.24, -4.72), "1%": (-2.58, -5.07)},
        9: {"10%": (-1.62, -4.26), "5%": (-1.95, -4.61), "2.5%": (-2.24, -4.89), "1%": (-2.58, -5.25)},
        10: {"10%": (-1.62, -4.42), "5%": (-1.95, -4.76), "2.5%": (-2.24, -5.06), "1%": (-2.58, -5.44)},
    },
    3: {
        0: {"10%": (-2.57, -2.57), "5%": (-2.86, -2.86), "2.5%": (-3.13, -3.13), "1%": (-3.43, -3.43)},
        1: {"10%": (-2.57, -2.91), "5%": (-2.86, -3.22), "2.5%": (-3.13, -3.5), "1%": (-3.43, -3.82)},
        2: {"10%": (-2.57, -3.21), "5%": (-2.86, -3.53), "2.5%": (-3.13, -3.8), "1%": (-3.43, -4.1)},
        3: {"10%": (-2.57, -3.46), "5%": (-2.86, -3.78), "2.5%": (-3.13, -4.05), "1%": (-3.43, -4.37)},
        4: {"10%": (-2.57, -3.66), "5%": (-2.86, -3.99), "2.5%": (-3.13, -4.26), "1%": (-3.43, -4.6)},
        5: {"10%": (-2.57, -3.86), "5%": (-2.86, -4.19), "2.5%": (-3.13, -4.46), "1%": (-3.43, -4.79)},
        6: {"10%": (-2.57, -4.04), "5%": (-2.86, -4.38), "2.5%": (-3.13, -4.66), "1%": (-3.43, -4.99)},
        7: {"10%": (-2.57, -4.23), "5%": (-2.86, -4.57), "2.5%": (-3.13, -4.85), "1%": (-3.43, -5.19)},
        8: {"10%": (-2.57, -4.4), "5%": (-2.86, -4.72), "2.5%": (-3.13, -5.02), "1%": (-3.43, -5.37)},
        9: {"10%": (-2.57, -4.56), "5%": (-2.86, -4.88), "2.5%": (-3.13, -5.18), "1%": (-3.42, -5.54)},
        10: {"10%": (-2.57, -4.69), "5%": (-2.86, -5.03), "2.5%": (-3.13, -5.34), "1%": (-3.43, -5.68)},
    },
    5: {
        0: {"10%": (-3.13, -3.13), "5%": (-3.41, -3.41), "2.5%": (-3.65, -3.66), "1%": (-3.96, -3.97)},
        1: {"10%": (-3.13, -3.4), "5%": (-3.41, -3.69), "2.5%": (-3.65, -3.96), "1%": (-3.96, -4.26)},
        2: {"10%": (-3.13, -3.63), "5%": (-3.41, -3.95), "2.5%": (-3.65, -4.2), "1%": (-3.96, -4.53)},
        3: {"10%": (-3.13, -3.84), "5%": (-3.41, -4.16), "2.5%": (-3.65, -4.42), "1%": (-3.96, -4.73)},
        4: {"10%": (-3.13, -4.04), "5%": (-3.41, -4.36), "2.5%": (-3.65, -4.62), "1%": (-3.96, -4.96)},
        5: {"10%": (-3.13, -4.21), "5%": (-3.41, -4.52), "2.5%": (-3.65, -4.79), "1%": (-3.96, -5.13)},
        6: {"10%": (-3.13, -4.37), "5%": (-3.41, -4.69), "2.5%": (-3.65, -4.96), "1%": (-3.96, -5.31)},
        7: {"10%": (-3.13, -4.53), "5%": (-3.41, -4.85), "2.5%": (-3.65, -5.14), "1%": (-3.96, -5.49)},
        8: {"10%": (-3.13, -4.68), "5%": (-3.41, -5.01), "2.5%": (-3.65, -5.3), "1%": (-3.96, -5.65)},
        9: {"10%": (-3.13, -4.82), "5%": (-3.41, -5.15), "2.5%": (-3.65, -5.44), "1%": (-3.96, -5.79)},
        10: {"10%": (-3.13, -4.96), "5%": (-3.41, -5.29), "2.5%": (-3.65, -5.59), "1%": (-3.96, -5.94)},
    },
}
