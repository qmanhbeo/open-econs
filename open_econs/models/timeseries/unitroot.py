from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd

from open_econs._version import __version__
from open_econs.core.call_capture import capture_call as _capture_call
from open_econs.models.timeseries.results import UnitRootResult

from arch.unitroot import ADF, DFGLS, KPSS, PhillipsPerron, ZivotAndrews


# ── bandwidth helpers (OE-side overrides of arch's hard-coded defaults) ──
# arch hardcodes Schwert's 1/4 exponent (12*(n/100)**0.25) for both ADF max_lags
# and PP's Newey-West bandwidth.  Stata uses a different exponent for PP's
# bandwidth: int(4*(T/100)**(2/9)).  We compute that ourselves and pass it in,
# since arch does not natively expose the Stata formula (standing rule 1: read
# the backend, then override deliberately).
def _stata_pp_bandwidth(nobs: int) -> int:
    return int(4 * (nobs / 100.0) ** (2.0 / 9.0))


def _schwert_max_lags(nobs: int) -> int:
    return int(math.ceil(12.0 * (nobs / 100.0) ** 0.25))


_LAG_SELECTION = ("fixed", "aic", "bic", "t-stat")


def adf(
    y: pd.Series | np.ndarray,
    *,
    lags: int = 0,
    trend: Literal["n", "c", "ct", "ctt"] = "c",
    lag_selection: Literal["fixed", "aic", "bic", "t-stat"] = "fixed",
    max_lags: int | None = None,
) -> UnitRootResult:
    """Augmented Dickey-Fuller unit-root test (wraps ``arch.unitroot.ADF``).

    Parameters
    ----------
    y : Series or ndarray
        The series to test.
    lags : int, default 0
        Number of augmentation lags. Only used when ``lag_selection="fixed"``.
        The default of 0 matches Stata ``dfuller``'s no-automatic-lag default.
    trend : {"n", "c", "ct", "ctt"}, default "c"
        Deterministic trend: none / constant / constant+trend /
        constant+trend+quadratic. Maps to Stata ``dfuller`` cases 1/2/4 and R
        ``ur.df`` types.
    lag_selection : {"fixed", "aic", "bic", "t-stat"}, default "fixed"
        Lag strategy. ``"fixed"`` uses ``lags`` exactly (Stata-equivalent).
        ``"aic"`` / ``"bic"`` / ``"t-stat"`` use ``arch``'s automatic
        lag selection with the Schwert max-lag ceiling (R/arch-equivalent).
    max_lags : int, optional
        Upper bound for automatic lag selection (defaults to arch's Schwert
        ``12*(n/100)**0.25``).

    Notes
    -----
    The displayed critical values are MacKinnon (2010) (arch's native table);
    Stata prints Fuller (1976) and R prints banded Fuller -- these diverge in
    small samples but share the same asymptotic test.  The MacKinnon (1994)
    p-value (``pvalue``) agrees with Stata's "MacKinnon approximate p-value"
    and is the project's primary parity anchor.  The CV vintage is labelled in
    :meth:`UnitRootResult.summary` (decision 1).  Legacy Fuller (1976)
    reproduction is tracked in FUTURE_WORK.
    """
    call = _capture_call(
        y=("series" if isinstance(y, pd.Series) else "array"),
        lags=lags, trend=trend, lag_selection=lag_selection, max_lags=max_lags,
    )
    s = pd.Series(y).astype(float).reset_index(drop=True)

    if lag_selection == "fixed":
        res = ADF(s.values, lags=int(lags), trend=trend)
    elif lag_selection in ("aic", "bic", "t-stat"):
        res = ADF(s.values, lags=None, trend=trend, max_lags=max_lags, method=lag_selection)
    else:
        raise ValueError(f"lag_selection must be one of {_LAG_SELECTION}")

    return UnitRootResult(
        test_name="Augmented Dickey-Fuller",
        stat=float(res.stat),
        pvalue=float(res.pvalue),
        critical_values=dict(res.critical_values),
        lags=int(res.lags),
        trend=trend,
        nobs=int(res.nobs),
        cv_vintage="MacKinnon (2010)",
        null_hypothesis=res.null_hypothesis,
        alternative_hypothesis=res.alternative_hypothesis,
        call=call,
    )


def pp(
    y: pd.Series | np.ndarray,
    *,
    trend: Literal["n", "c", "ct"] = "c",
    test_type: Literal["tau", "rho"] = "tau",
    bandwidth: Literal["stata", "schwert", "fixed"] = "stata",
    lags: int | None = None,
) -> UnitRootResult:
    """Phillips-Perron unit-root test (wraps ``arch.unitroot.PhillipsPerron``).

    Parameters
    ----------
    y : Series or ndarray
        The series to test.
    trend : {"n", "c", "ct"}, default "c"
        Deterministic trend.
    test_type : {"tau", "rho"}, default "tau"
        ``"tau"`` = Z(t) test (Stata ``pperron`` default); ``"rho"`` = Z(rho).
    bandwidth : {"stata", "schwert", "fixed"}, default "stata"
        Newey-West bandwidth for the long-run variance.

        - ``"stata"`` : ``int(4*(T/100)**(2/9))`` -- matches Stata ``pperron``'s
          default (OE-side override of arch's hard-coded Schwert 1/4 exponent).
        - ``"schwert"`` : ``int(12*(n/100)**0.25)`` -- arch/R default.
        - ``"fixed"`` : use the explicit ``lags`` value.

    Notes
    -----
    arch's default ``lags`` (NW bandwidth) is Schwert ``12*(n/100)**0.25``; we
    override to Stata's ``4*(n/100)**(2/9)`` by default per decision 3.  CV
    vintage is MacKinnon (2010) (arch reuses the ADF tables); Stata prints
    Fuller (1976); parity is asserted on the MacKinnon (1994) p-value (decision
    2).
    """
    call = _capture_call(
        y=("series" if isinstance(y, pd.Series) else "array"),
        trend=trend, test_type=test_type, bandwidth=bandwidth, lags=lags,
    )
    s = pd.Series(y).astype(float).reset_index(drop=True)
    n = len(s)

    if bandwidth == "stata":
        nw = _stata_pp_bandwidth(n)
    elif bandwidth == "schwert":
        nw = int(math.ceil(12.0 * (n / 100.0) ** 0.25))
    elif bandwidth == "fixed":
        if lags is None:
            raise ValueError("bandwidth='fixed' requires an explicit lags= value")
        nw = int(lags)
    else:
        raise ValueError("bandwidth must be 'stata', 'schwert', or 'fixed'")

    res = PhillipsPerron(s.values, lags=nw, trend=trend, test_type=test_type)
    return UnitRootResult(
        test_name=f"Phillips-Perron ({'Z-'+test_type})",
        stat=float(res.stat),
        pvalue=float(res.pvalue),
        critical_values=dict(res.critical_values),
        lags=int(res.lags),
        trend=trend,
        nobs=int(res.nobs),
        cv_vintage="MacKinnon (2010, ADF tables)",
        null_hypothesis=res.null_hypothesis,
        alternative_hypothesis=res.alternative_hypothesis,
        call=call,
    )


def kpss(
    y: pd.Series | np.ndarray,
    *,
    trend: Literal["c", "ct"] = "c",
    lags: int | None = None,
) -> UnitRootResult:
    """KPSS stationarity test (wraps ``arch.unitroot.KPSS``).

    Parameters
    ----------
    y : Series or ndarray
        The series to test.
    trend : {"c", "ct"}, default "c"
        "c" = level stationarity, "ct" = trend stationarity.
    lags : int, optional
        Bandwidth for the long-run variance.  If None, arch's data-dependent
        Hobijn et al. (1998) rule is used (arch's default).

    Notes
    -----
    arch reports Hobijn et al. (2004) critical values from its own simulation;
    R ``ur.kpss`` reports Kwiatkowski et al. (1992).  These are numerically
    close (e.g. level 5%: 0.4614 vs 0.463) but from different sources.  The CV
    vintage is labelled in :meth:`UnitRootResult.summary` (decision 1).  KPSS
    has no Stata base command (SSC community only), so its parity anchor is R
    (urca).
    """
    call = _capture_call(
        y=("series" if isinstance(y, pd.Series) else "array"),
        trend=trend, lags=lags,
    )
    s = pd.Series(y).astype(float).reset_index(drop=True)
    res = KPSS(s.values, lags=lags, trend=trend)
    return UnitRootResult(
        test_name="KPSS Stationarity",
        stat=float(res.stat),
        pvalue=float(res.pvalue),
        critical_values=dict(res.critical_values),
        lags=int(res.lags),
        trend=trend,
        nobs=int(res.nobs),
        cv_vintage="Hobijn et al. (2004)",
        null_hypothesis=res.null_hypothesis,
        alternative_hypothesis=res.alternative_hypothesis,
        call=call,
    )


def dfgls(
    y: pd.Series | np.ndarray,
    *,
    trend: Literal["c", "ct"] = "c",
    lags: int | None = None,
    max_lags: int | None = None,
    method: Literal["aic", "bic", "t-stat"] = "aic",
) -> UnitRootResult:
    """DF-GLS (ERS) unit-root test (wraps ``arch.unitroot.DFGLS``).

    Parameters
    ----------
    y : Series or ndarray
        The series to test.
    trend : {"c", "ct"}, default "c"
        "c" = constant (ERS cbar=-7.0), "ct" = constant+trend (cbar=-13.5).
    lags : int, optional
        Exact number of augmentation lags.  If None, automatic selection is
        used (``method``).
    max_lags : int, optional
        Max lag for automatic selection (defaults to Schwert
        ``12*(n/100)**0.25``, matching Stata ``dfgls``'s default maxlag).
    method : {"aic", "bic", "t-stat"}, default "aic"
        Automatic lag-selection criterion.

    Notes
    -----
    arch's default lag selection is AIC on the OLS-detrended series; Stata
    ``dfgls`` uses Ng-Perron sequential-t / SIC / MAIC.  The GLS detrending
    (ERS cbar) matches exactly, and so does the default max-lag ceiling, but the
    lag-selection *method* differs.  The Ng-Perron port is tracked in
    FUTURE_WORK; the arch AIC default is used here (decision 3 fallback).
    CV vintage is MacKinnon-dfgls (arch's own simulation), which differs from
    Stata's ERS (1996) / Fuller / Cheung-Lai (1995) table.
    """
    call = _capture_call(
        y=("series" if isinstance(y, pd.Series) else "array"),
        trend=trend, lags=lags, max_lags=max_lags, method=method,
    )
    s = pd.Series(y).astype(float).reset_index(drop=True)
    res = DFGLS(s.values, lags=lags, trend=trend, max_lags=max_lags, method=method)
    return UnitRootResult(
        test_name="Dickey-Fuller GLS (ERS)",
        stat=float(res.stat),
        pvalue=float(res.pvalue),
        critical_values=dict(res.critical_values),
        lags=int(res.lags),
        trend=trend,
        nobs=int(res.nobs),
        cv_vintage="MacKinnon (DF-GLS simulation)",
        null_hypothesis=res.null_hypothesis,
        alternative_hypothesis=res.alternative_hypothesis,
        call=call,
    )


def zivot_andrews(
    y: pd.Series | np.ndarray,
    *,
    trend: Literal["c", "t", "ct"] = "c",
    lags: int = 0,
    lag_selection: Literal["fixed", "aic", "bic", "t-stat"] = "fixed",
    max_lags: int | None = None,
) -> UnitRootResult:
    """Zivot-Andrews unit-root test with a single structural break.

    Parameters
    ----------
    y : Series or ndarray
        The series to test.
    trend : {"c", "t", "ct"}, default "c"
        Break model: constant / trend / constant+trend.
    lags : int, default 0
        Augmentation lags. Used when ``lag_selection="fixed"``.  The default of
        0 matches R ``ur.za`` (``lag=NULL`` -> 0), the stronger anchor since
        Stata has no base ZA command.
    lag_selection : {"fixed", "aic", "bic", "t-stat"}, default "fixed"
        Lag strategy. ``"fixed"`` uses ``lags`` exactly (R-equivalent);
        ``"aic"`` etc. use arch's automatic selection (arch's default).
    max_lags : int, optional
        Upper bound for automatic lag selection.

    Notes
    -----
    arch reports its own Monte-Carlo critical values (100k reps); R ``ur.za``
    reports Zivot-Andrews (1992) asymptotic values.  These differ and are
    labelled in :meth:`UnitRootResult.summary` (decision 1).  ZA has no Stata
    base command, so its parity anchor is R (urca).
    """
    call = _capture_call(
        y=("series" if isinstance(y, pd.Series) else "array"),
        trend=trend, lags=lags, lag_selection=lag_selection, max_lags=max_lags,
    )
    s = pd.Series(y).astype(float).reset_index(drop=True)

    if lag_selection == "fixed":
        res = ZivotAndrews(s.values, lags=int(lags), trend=trend)
    elif lag_selection in ("aic", "bic", "t-stat"):
        res = ZivotAndrews(
            s.values, lags=None, trend=trend, max_lags=max_lags, method=lag_selection
        )
    else:
        raise ValueError(f"lag_selection must be one of {_LAG_SELECTION}")

    return UnitRootResult(
        test_name="Zivot-Andrews",
        stat=float(res.stat),
        pvalue=None,  # arch's ZA does not compute a p-value
        critical_values=dict(res.critical_values),
        lags=int(res.lags),
        trend=trend,
        nobs=int(res.nobs),
        cv_vintage="arch Monte Carlo (100k reps)",
        null_hypothesis=res.null_hypothesis,
        alternative_hypothesis=res.alternative_hypothesis,
        call=call,
    )
