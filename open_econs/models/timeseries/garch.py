from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from open_econs.core.call_capture import capture_call as _capture_call
from open_econs.models.timeseries.results import GARCHResult

from arch import arch_model
from arch.univariate.base import ARCHModelResult


def garch(
    y: pd.Series | np.ndarray,
    *,
    p: int = 1,
    q: int = 1,
    o: int = 0,
    mean: Literal["constant", "zero", "ar", "arma", "harx"] = "constant",
    lags: int | None = None,
    vol: Literal["GARCH", "EGARCH", "GJR", "ARCH", "HARCH", "Constant"] = "GARCH",
    dist: Literal["normal", "t", "skewt", "ged"] = "normal",
    power: float = 2.0,
    **fit_kwargs: Any,
) -> GARCHResult:
    """GARCH-family volatility model (wraps ``arch.arch_model``).

    Parameters
    ----------
    y : Series or ndarray
        The (typically mean-zero or mean-modelled) series to model.
    p : int, default 1
        Order of the symmetric GARCH term.  Always passed explicitly so OE
        never silently relies on arch's (1,1) default -- this keeps OE explicit
        against Stata, which requires ``arch()``/``garch()`` orders to be stated.
    q : int, default 1
        Order of the ARCH term.
    o : int, default 0
        Order of the asymmetry (leverage) term (GJR/EGARCH).
    mean : {"constant", "zero", "ar", "arma", "harx"}, default "constant"
        Mean model.
    lags : int, optional
        Lag(s) for an ``"ar"`` / ``"arma"`` mean model.
    vol : {"GARCH", "EGARCH", "GJR", "ARCH", "HARCH", "Constant"}, default "GARCH"
        Volatility process.
    dist : {"normal", "t", "skewt", "ged"}, default "normal"
        Error distribution.  Defaults to Normal, matching Stata ``arch`` and
        ``rugarch``.
    power : float, default 2.0
        Power of the GARCH representation (2 = standard GARCH; 1 = absolute
        value / Taylor/Schwert).

    Notes
    -----
    No variance-targeting divergence: ``arch`` (confirmed: omega is
    parameter[0], freely estimated), Stata ``arch`` + ``garch()``, and
    ``rugarch`` all estimate the variance constant freely via full MLE and
    default to Gaussian errors.  The only divergence is the GARCH lag default
    (arch/rugarch = (1,1); Stata requires explicit ``garch(1)``), reconciled by
    always passing ``p``/``q`` explicitly (decision: GARCH is the lowest-risk
    v1.1.0 item).
    """
    call = _capture_call(
        y=("series" if isinstance(y, pd.Series) else "array"),
        p=p, q=q, o=o, mean=mean, lags=lags, vol=vol, dist=dist, power=power,
        **fit_kwargs,
    )
    s = pd.Series(y).astype(float).reset_index(drop=True)

    am = arch_model(
        s.values,
        mean=mean,
        lags=lags,
        vol=vol,
        p=p,
        o=o,
        q=q,
        power=power,
        dist=dist,
        rescale=False,
    )
    fit_kwargs.setdefault("disp", "off")
    res: ARCHModelResult = am.fit(**fit_kwargs)

    params = res.params
    std_errors = res.std_err
    t_stats = res.tvalues
    p_values = res.pvalues
    conf_int = res.conf_int()

    return GARCHResult(
        params=params,
        std_errors=std_errors,
        t_stats=t_stats,
        p_values=p_values,
        conf_int=conf_int,
        llf=float(res.loglikelihood),
        aic=float(res.aic),
        bic=float(res.bic),
        nobs=int(res.nobs),
        residuals=pd.Series(res.resid, name="resid"),
        conditional_volatility=pd.Series(res.conditional_volatility, name="cond_vol"),
        call=call,
    )
