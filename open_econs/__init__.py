from ._version import __version__
from .models.linear.ols import ols
from .models.decomposition.oaxaca import oaxaca
from .models.discrete.logit import logit
from .models.discrete.probit import probit
from .models.linear.fe import fe
from .models.linear.iv import iv
from .models.causal.did import did, event_study
from .models.causal.balance import balance
from .core.context import Context

reg = ols

__all__ = [
    "ols", "reg", "logit", "probit", "fe", "iv", "oaxaca",
    "did", "event_study", "balance",
    "Context", "__version__",
]