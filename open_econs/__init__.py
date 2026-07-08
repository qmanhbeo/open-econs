from ._version import __version__
from .models.linear.ols import ols
from .models.decomposition.oaxaca import oaxaca
from .core.context import Context

reg = ols

__all__ = ["ols", "reg", "oaxaca", "Context", "__version__"]