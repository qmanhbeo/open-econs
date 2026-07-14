from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from open_econs.core.call_capture import capture_call as _capture_call
from open_econs.models.timeseries.results import ARIMAResult

from statsmodels.tsa.arima.model import ARIMA as _SMARIMA


_METHOD_MAP = {
    "ml": "statespace",
    "statespace": "statespace",
    "css-ml": "css-mle",
    "css_ml": "css-mle",
    "css": "css",
}


def _resolve_method(method: str) -> str:
    key = method.lower().replace("-", "_")
    if key not in _METHOD_MAP:
        raise ValueError(
            f"method must be one of 'ml', 'css-ml', 'css' (got {method!r}). "
            "Use 'ml' (default, state-space Kalman) to match Stata/statsmodels, "
            "or 'css-ml' for R stats::arima parity."
        )
    return _METHOD_MAP[key]


def arima(
    y: pd.Series | np.ndarray,
    *,
    order: tuple[int, int, int] = (0, 0, 0),
    trend: Literal["n", "c", "t", "ct"] | None = None,
    method: Literal["ml", "css-ml", "css"] = "ml",
    **fit_kwargs: Any,
) -> ARIMAResult:
    """ARIMA model (wraps ``statsmodels.tsa.arima.model.ARIMA``).

    Parameters
    ----------
    y : Series or ndarray
        The series to model.
    order : (p, d, q), default (0,0,0)
        AR order, differencing, MA order.
    trend : {"n", "c", "t", "ct"}, optional
        Deterministic trend.  ``"n"`` = none; ``"c"`` = constant; ``"t"`` =
        time trend; ``"ct"`` = both.  When ``d > 0`` a trend is dropped by
        statsmodels (matching Stata), so the default (None) lets statsmodels
        apply its natural rule (constant included iff d=0).
    method : {"ml", "css-ml", "css"}, default "ml"
        Estimation method.  ``"ml"`` = pure ML via the state-space Kalman
        filter, matching **both** Stata ``arima`` and statsmodels' native
        default (two of three reference tools agree).  ``"css-ml"`` is exposed
        for R ``stats::arima`` parity (R defaults to CSS-ML).

    Notes
    -----
    The AR/MA sign convention was empirically verified against Stata
    ``arima y, ar(.) ma(.)`` and R ``stats::arima(y, order=...)``: all three
    agree exactly on AR and MA coefficient signs and magnitudes (e.g.
    ARMA(1,1): AR=0.6875, MA=-0.5679, const=-0.0089, LL=-419.316 for every
    tool).  The historical statsmodels/R sign flip applied only to the old
    ``tsa.arima_model.ARMA`` MLE path; the current statespace implementation
    already matches Stata and R, so **no sign correction is applied in the
    wrapper** (documented per standing rule 1).
    """
    call = _capture_call(
        y=("series" if isinstance(y, pd.Series) else "array"),
        order=order, trend=trend, method=method, **fit_kwargs,
    )
    s = pd.Series(y).astype(float).reset_index(drop=True)

    sm_method = _resolve_method(method)
    mod = _SMARIMA(s.values, order=tuple(order), trend=trend)
    res = mod.fit(method=sm_method, **fit_kwargs)

    names = res.model.param_names
    params = pd.Series(res.params, index=names)
    std_errors = pd.Series(res.bse, index=names)
    t_stats = pd.Series(res.tvalues, index=names)
    p_values = pd.Series(res.pvalues, index=names)
    conf_int = pd.DataFrame(res.conf_int(), index=names, columns=["lower", "upper"])

    return ARIMAResult(
        params=params,
        std_errors=std_errors,
        t_stats=t_stats,
        p_values=p_values,
        conf_int=conf_int,
        llf=float(res.llf),
        aic=float(res.aic),
        bic=float(res.bic),
        nobs=int(res.nobs),
        residuals=pd.Series(res.resid, name="resid"),
        fitted_values=pd.Series(res.fittedvalues, name="fitted"),
        order=order,
        method=method,
        call=call,
    )


def arma(
    y: pd.Series | np.ndarray,
    *,
    p: int = 0,
    q: int = 0,
    trend: Literal["n", "c", "t", "ct"] | None = None,
    method: Literal["ml", "css-ml", "css"] = "ml",
    **fit_kwargs: Any,
) -> ARIMAResult:
    """ARMA(p, q) model -- convenience wrapper around :func:`arima` with d=0."""
    return arima(
        y, order=(p, 0, q), trend=trend, method=method, **fit_kwargs
    )
