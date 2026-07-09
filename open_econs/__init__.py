from ._version import __version__
from .models.linear.ols import ols
from .models.decomposition.oaxaca import oaxaca
from .models.discrete.logit import logit
from .models.discrete.probit import probit
from .models.linear.fe import fe
from .models.linear.iv import iv
from .core.context import Context

reg = ols

__all__ = ["ols", "reg", "logit", "probit", "fe", "iv", "oaxaca", "Context", "__version__"]