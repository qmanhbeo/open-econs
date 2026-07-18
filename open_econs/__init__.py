from __future__ import annotations

import warnings as _warnings
from typing import Any

from ._version import __version__
from ._internal.errors import VcovTypeNotSupportedError
from .models.linear.ols import ols
from .models.decomposition.oaxaca import oaxaca
from .models.discrete.logit import logit
from .models.discrete.probit import probit
from .models.discrete.mlogit import mlogit
from .models.linear.fe import fe
from .models.linear.iv import iv
from .models.linear.abond import abond
from .models.linear.gmm import gmm, GMMResult
from .models.nonlinear.nls import nls, NLSResult
from .models.causal.did import did, event_study
from .models.causal.balance import balance
from .models.causal.did_cs import did_cs, CsDiDResult, AggteResult
from .models.causal.did_gardner import did_gardner, GardnerResult
from .models.causal.did_sa import did_sa, SaDiDResult
from .models.causal.cem import cem
from .models.causal.psm import psm
from .models.causal.sensitivity import rosenbaum_bounds
from .models.causal.rdd import density_test, rdd
from .models.causal.synth import synth, SynthResult
from .models.causal.placebo import placebo_space, placebo_time
from .core.context import Context
from .core.panel_context import PanelContext
from .models.timeseries import (
    TimeSeriesContext,
    adf,
    ardl_fit,
    ardl_select_order,
    arma,
    arima,
    bounds_test,
    dfgls,
    garch,
    granger_causality,
    instantaneous_causality,
    johansen_cointegration,
    kpss,
    pp,
    uecm_fit,
    var_fit,
    var_select_order,
    vec2var,
    vecm_fit,
    zivot_andrews,
)

reg = ols


def staggered_did(*args: Any, **kwargs: Any) -> object:
    _warnings.warn(
        "`staggered_did` is deprecated; use `did_cs` instead.",
        FutureWarning, stacklevel=2,
    )
    return did_cs(*args, **kwargs)


def did_sun_abraham(*args: Any, **kwargs: Any) -> object:
    _warnings.warn(
        "`did_sun_abraham` is deprecated; use `did_sa` instead.",
        FutureWarning, stacklevel=2,
    )
    return did_sa(*args, **kwargs)


__all__ = [
    "ols", "reg", "logit", "probit", "mlogit", "fe", "iv", "oaxaca",
    "did", "event_study", "balance", "abond", "did_cs", "density_test", "cem",
    "psm", "rdd", "rosenbaum_bounds", "gmm", "GMMResult",
    "nls", "NLSResult",
    "synth", "SynthResult", "AggteResult", "CsDiDResult", "did_gardner", "GardnerResult",
    "did_sa", "SaDiDResult",
    "staggered_did", "did_sun_abraham",
    "placebo_space", "placebo_time",
    "VcovTypeNotSupportedError",
    "Context", "PanelContext", "TimeSeriesContext", "__version__",
    "adf", "pp", "kpss", "dfgls", "zivot_andrews",
    "garch", "arima", "arma",
    "var_fit", "var_select_order", "johansen_cointegration",
    "granger_causality", "instantaneous_causality",
    "vecm_fit", "vec2var",
    "ardl_fit", "uecm_fit", "ardl_select_order", "bounds_test",
]