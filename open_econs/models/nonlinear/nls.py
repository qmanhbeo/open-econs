"""Nonlinear least squares (NLS) estimator (public API).

Single-equation nonlinear mean function estimated by Gauss-Newton via
:func:`scipy.optimize.least_squares`.  The RHS of the formula is parsed with
``sympy.parse_expr``; symbols that appear in ``start_values`` are treated as
parameters (and estimated), symbols that appear as columns in *data* are
treated as data, and anything else is an error.  The Jacobian is differentiated
analytically with ``sympy.diff`` and passed to ``least_squares``; if analytic
differentiation fails for any parameter the solver falls back to its built-in
numerical Jacobian (and the result records ``jacobian_method="numerical"``).

Standard errors follow the library's existing convention:

* ``cov_type="nonrobust"`` -- naive Gauss-Newton variance ``sigma^2 (J'J)^-1``
  with ``sigma^2 = SSR / (n - k)`` (matches :func:`scipy.optimize.curve_fit`).
* ``cov_type`` in ``{"HC0","HC1","HC2","HC3"}`` -- heteroskedasticity-robust
  MacKinnon-White sandwich (see :func:`open_econs.core.cov.white_cov`).
* ``cov_type="cluster"`` -- multi-way cluster-robust via
  :func:`open_econs.core.cov.multiway_cluster_cov` with the solution Jacobian
  playing the role of the design matrix.
* ``cov_type="HAC"`` -- Newey-West HAC via
  :func:`open_econs.core.cov.newey_west_cov`.

This is new, additive code only: no existing estimator is modified.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm as _norm
from sympy import diff as _sym_diff
from sympy import lambdify as _sym_lambdify
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

from open_econs._internal import errors
from open_econs._version import __version__
from open_econs.core.base import BaseModel
from open_econs.core.call_capture import capture_call as _capture_call
from open_econs.core.cov import white_cov


class NLSResult(BaseModel):
    """Result of a nonlinear least-squares estimation.

    Immutable result with the uniform interface (``.tidy()``, ``.summary()``,
    ``.export()``, ``.vcov()``, ``.to_latex()`` / ``.to_html()``).  Adds
    convergence diagnostics (``.success``, ``.n_function_evaluations``,
    ``.cost``) and ``jacobian_method`` (``"analytic"`` or ``"numerical"``)
    so the estimation path is auditable rather than silent.

    Coefficients and standard errors are returned as named ``pd.Series`` keyed
    by the parameter names (the keys of ``start_values``).
    """

    def __init__(
        self,
        *,
        formula: str,
        rhs_formula: str,
        coefficients: pd.Series,
        std_errors: pd.Series,
        t_stats: pd.Series,
        p_values: pd.Series,
        conf_int: pd.DataFrame,
        cov_type: str,
        n_obs: int,
        jacobian_method: str,
        success: bool,
        n_function_evaluations: int,
        n_jacobian_evaluations: int,
        cost: float,
        status: int,
        message: str,
        optimality: float,
        vcov_matrix: np.ndarray,
        fitted_values: pd.Series,
        residuals: pd.Series,
        call: dict[str, Any],
    ) -> None:
        self.formula = formula
        self.rhs_formula = rhs_formula
        self.data_shape = (n_obs, len(coefficients))
        self.cov_type = cov_type
        self.call = call
        self.timestamp = datetime.now()
        self.package_version = __version__

        self.coefficients = coefficients
        self.std_errors = std_errors
        self.t_stats = t_stats
        self.p_values = p_values
        self.conf_int = conf_int
        self.n_obs = n_obs
        self.jacobian_method = jacobian_method

        # Convergence diagnostics straight from scipy.optimize.least_squares'
        # OptimizeResult.  NOTE: least_squares exposes NO iteration count field
        # (no ``nit``); the closest proxies are the function/Jacobian
        # evaluation counts, which we surface directly and via the
        # ``n_iterations`` convenience property.
        self.success = bool(success)
        self.n_function_evaluations = int(n_function_evaluations)
        self.n_jacobian_evaluations = int(n_jacobian_evaluations)
        self.cost = float(cost)
        self.status = int(status)
        self.message = str(message)
        self.optimality = float(optimality)

        self._vcov = np.asarray(vcov_matrix, dtype=float)
        self.fitted_values = fitted_values
        self.residuals = residuals

        self._freeze()

    @property
    def n_iterations(self) -> int:
        """Function-evaluation count from the solver.

        ``scipy.optimize.least_squares`` does not expose a dedicated iteration
        counter (no ``nit`` attribute), so this returns ``nfev`` -- the number
        of residual-function evaluations -- which is the honest proxy for
        "how much work the solver did".
        """
        return self.n_function_evaluations

    def tidy(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "Variable": self.coefficients.index,
            "Coef": self.coefficients.values,
            "Std Err": self.std_errors.values,
            "t": self.t_stats.values,
            "P>|t|": self.p_values.values,
            "0.025": self.conf_int["lower"].values,
            "0.975": self.conf_int["upper"].values,
        })
        df.index.name = None
        return df

    def summary(self) -> str:
        header = (
            f"                  Nonlinear Least Squares Results                        \n"
            f"======================================================================\n"
            f"Dep. Variable:               {self.formula.split('~')[0].strip()}\n"
            f"No. Observations:            {self.n_obs}\n"
            f"Covariance Type:             {self.cov_type}\n"
            f"Jacobian:                    {self.jacobian_method}\n"
            f"Method:                      Gauss-Newton (scipy least_squares)\n"
            f"Converged:                   {self.success}\n"
            f"Function Evaluations:        {self.n_function_evaluations}\n"
            f"Jacobian Evaluations:        {self.n_jacobian_evaluations}\n"
            f"Cost (0.5*SSR):              {self.cost:.6f}\n"
            f"Optimality (inf-norm grad):  {self.optimality:.6e}\n"
            f"Status:                      {self.status} ({self.message.strip()})\n"
            f"======================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return header + tbl + "\n======================================================================\n"

    def vcov(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._vcov,
            index=self.coefficients.index,
            columns=self.coefficients.index,
        )

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["jacobian_method"] = self.jacobian_method
        d["convergence"] = {
            "success": self.success,
            "n_function_evaluations": self.n_function_evaluations,
            "n_jacobian_evaluations": self.n_jacobian_evaluations,
            "cost": self.cost,
            "status": self.status,
            "optimality": self.optimality,
            "message": self.message,
        }
        return d
