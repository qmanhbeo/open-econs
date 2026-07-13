from ._version import __version__
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
from .models.causal.staggered_did import staggered_did
from .models.causal.cem import cem
from .models.causal.psm import psm
from .models.causal.sensitivity import rosenbaum_bounds
from .models.causal.rdd import density_test, rdd
from .models.causal.synth import synth, SynthResult
from .models.causal.placebo import placebo_space, placebo_time
from .core.context import Context
from .core.panel_context import PanelContext

reg = ols

__all__ = [
    "ols", "reg", "logit", "probit", "mlogit", "fe", "iv", "oaxaca",
    "did", "event_study", "balance", "abond", "staggered_did", "density_test", "cem",
    "psm", "rdd", "rosenbaum_bounds", "gmm", "GMMResult",
    "nls", "NLSResult",
    "synth", "SynthResult",
    "placebo_space", "placebo_time",
    "Context", "PanelContext", "__version__",
]