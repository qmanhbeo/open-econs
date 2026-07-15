from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from open_econs.core.call_capture import capture_call as _capture_call
from open_econs.models.timeseries.results import GARCHResult

from arch import arch_model
from arch.univariate.base import ARCHModelResult


# arch 8.0.0 renamed several model identifiers and dropped standalone spellings:
#   * GJR-GARCH is selected via `vol="GARCH"` with `o > 0` (no `vol="GJR"` name).
#   * no ARMA mean model exists (only Constant/Zero/AR/ARX/HAR/HARX/LS).
# arch's type stubs only list the Capital spellings, so we keep open_econs's
# lowercase API and map to arch's canonical identifiers at the call site.
_MEAN_MAP: dict[str, Literal["Constant", "Zero", "LS", "AR", "ARX", "HAR", "HARX", "constant", "zero"]] = {
    "constant": "Constant",
    "zero": "Zero",
    "ar": "AR",
    "harx": "HARX",
}
_VOL_MAP: dict[str, Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"]] = {
    "GARCH": "GARCH",
    "EGARCH": "EGARCH",
    "ARCH": "ARCH",
    "HARCH": "HARCH",
}


def garch(
    y: pd.Series | np.ndarray,
    *,
    p: int = 1,
    q: int = 1,
    o: int = 0,
    mean: Literal["constant", "zero", "ar", "harx"] = "constant",
    lags: int | None = None,
    vol: Literal["GARCH", "EGARCH", "ARCH", "HARCH"] = "GARCH",
    dist: Literal["normal", "t", "skewt", "ged"] = "normal",
    power: float = 2.0,
    backcast: Literal["stata", "arch"] | float = "stata",
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
    mean : {"constant", "zero", "ar", "harx"}, default "constant"
        Mean model.
    lags : int, optional
        Lag(s) for an ``"ar"`` / ``"harx"`` mean model.
    vol : {"GARCH", "EGARCH", "ARCH", "HARCH"}, default "GARCH"
        Volatility process.  GJR-GARCH is obtained with ``vol="GARCH"`` and
        ``o > 0`` (arch 8.0.0 folded the standalone ``"GJR"`` spelling into the
        asymmetric GARCH model).
    dist : {"normal", "t", "skewt", "ged"}, default "normal"
        Error distribution.  Defaults to Normal, matching Stata ``arch`` and
        ``rugarch``.
    power : float, default 2.0
        Power of the GARCH representation (2 = standard GARCH; 1 = absolute
        value / Taylor/Schwert).
    backcast : {"stata", "arch"} or float, default "stata"
        Presample variance convention.  ``"stata"`` uses the sample mean of
        squared residuals (``mean(e²)``), matching Stata ``arch0(xb)`` (default)
        and R ``rugarch`` ``rec.init="all"``.  ``"arch"`` uses arch's native
        exponentially-weighted average of the first 75 squared residuals
        (decay 0.94).  A float is used directly as sigma²₀.

    Notes
    -----
    No variance-targeting divergence: ``arch`` (confirmed: omega is
    parameter[0], freely estimated), Stata ``arch`` + ``garch()``, and
    ``rugarch`` all estimate the variance constant freely via full MLE and
    default to Gaussian errors.  The only divergence is the GARCH lag default
    (arch/rugarch = (1,1); Stata requires explicit ``garch(1)``), reconciled by
    always passing ``p``/``q`` explicitly (decision: GARCH is the lowest-risk
    v1.1.0 item).

    The presample variance (backcast) convention diverges across tools:
    Stata/R use ``mean(e²)``; arch uses an EWMA of the first 75 observations.
    OE defaults to the Stata/R convention for parity.  The remaining
    coefficient-level gap (~1e-3 relative on beta) is the omega-beta ridge
    (near-collinearity in the variance recursion), a genuine flat-likelihood
    identifiability issue that no backcast choice can close.
    """
    call = _capture_call(
        y=("series" if isinstance(y, pd.Series) else "array"),
        p=p, q=q, o=o, mean=mean, lags=lags, vol=vol, dist=dist, power=power,
        backcast=backcast,
        **fit_kwargs,
    )
    s = pd.Series(y).astype(float).reset_index(drop=True)

    am = arch_model(
        s.values,
        mean=_MEAN_MAP[mean],
        lags=lags,
        vol=_VOL_MAP[vol],
        p=p,
        o=o,
        q=q,
        power=power,
        dist=dist,
        rescale=False,
    )
    fit_kwargs.setdefault("disp", "off")

    if backcast == "stata":
        pre_res = am.fit(**fit_kwargs)
        bc_val = float(np.mean(pre_res.resid**2))
        fit_kwargs["backcast"] = bc_val
    elif backcast == "arch":
        pass  # arch's default EWMA backcast
    else:
        fit_kwargs["backcast"] = float(backcast)

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
