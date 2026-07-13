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
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm as _norm

try:
    from sympy import diff as _sym_diff
    from sympy import lambdify as _sym_lambdify
    from sympy.parsing.sympy_parser import (
        convert_xor,
        parse_expr,
        standard_transformations,
    )

    _HAVE_SYMPY = True
except ImportError:
    _HAVE_SYMPY = False

from open_econs._internal import errors
from open_econs._version import __version__
from open_econs.core.base import BaseModel
from open_econs.core.call_capture import capture_call as _capture_call
from open_econs.core.cov import white_cov
from open_econs.core.cov_type import validate_cov_type


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
        """R-broom-style coefficient table (Variable, Coef, Std Err, t, P>|t|, CI)."""
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
        """Pretty-printed terminal summary of the NLS fit (incl. convergence stats)."""
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
        """Return the parameter variance-covariance matrix as a DataFrame."""
        return pd.DataFrame(
            self._vcov,
            index=self.coefficients.index,
            columns=self.coefficients.index,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable dict; extends the base payload with NLS convergence fields."""
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


def nls(
    formula: str,
    data: pd.DataFrame,
    start_values: dict[str, float],
    *,
    cov_type: str = "HC2",
    cluster: str | list[str] | None = None,
    max_lags: int | None = None,
    time: str | None = None,
    **solver_kwargs: Any,
) -> NLSResult:
    """Estimate a nonlinear least-squares model.

    Parameters
    ----------
    formula : str
        Two-sided formula ``"y ~ <nonlinear expression>"``, e.g.
        ``"y ~ a * exp(-b * x) + c"`` or ``"y ~ beta0 + beta1 * x ** beta2"``.
        The RHS is parsed by ``sympy.parse_expr`` (``^`` is treated as power).
    data : pd.DataFrame
        Data containing the dependent variable and every data column named in
        the RHS expression.
    start_values : dict[str, float]
        Starting values for every *parameter* in the model.  A symbol in the
        RHS is a parameter if and only if its name is a key here.  Parameter
        names become the rows of the coefficient/SE series, in this dict's
        insertion order.
    cov_type : {"nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster", "HAC"}, default "HC2"
        Covariance estimator.  ``"HC2"`` (default) matches the library's
        standard robust SE.  ``"cluster"`` requires *cluster*; ``"HAC"``
        requires *max_lags*.
    cluster : str or list of str, optional
        Column name(s) for cluster-robust standard errors.  Required (and only
        used) when ``cov_type="cluster"``.
    max_lags : int, optional
        Number of lags for Newey-West HAC (required when ``cov_type="HAC"``).
    time : str, optional
        Column with the time index used to order observations for Newey-West
        HAC (or the panel time id when combined with ``cluster``).  Ignored
        unless ``cov_type="HAC"``; when omitted, HAC uses the data's current
        row order.  Mirrors :func:`open_econs.ols`'s ``time=`` for HAC.
    **solver_kwargs
        Extra keyword arguments forwarded verbatim to
        :func:`scipy.optimize.least_squares` (e.g. ``method``, ``bounds``,
        ``xtol``, ``ftol``, ``max_nfev``).  ``jac`` is set internally and must
        not be passed here.

    Returns
    -------
    NLSResult
        Immutable result with named coefficient/SE series, ``.vcov()``,
        convergence diagnostics, and ``.jacobian_method``.

    Notes
    -----
    **Parameter vs data disambiguation.**  A symbol in the RHS is a *parameter*
    if its name is a key in ``start_values``; it is *data* if it is a column in
    ``data``.  If a name is both, or neither, a clear ``ValueError`` is raised
    -- the estimator never guesses which one wins.

    **Analytic Jacobian.**  The gradient of the mean function w.r.t. each
    parameter is computed with ``sympy.diff`` and passed to ``least_squares``.
    If differentiation or lambdification fails for *any* parameter, the whole
    model falls back to ``least_squares``'s numerical Jacobian and the result
    records ``jacobian_method="numerical"`` (never silent).
    """
    if not _HAVE_SYMPY:
        raise ImportError(
            "nls() requires the `sympy` package (pip install open-econs[nls])."
        )

    cov_type = validate_cov_type(
        cov_type,
        accepted={"nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster", "HAC"},
        estimator="nls()",
    )

    # ── covariance / cluster / HAC argument validation ──────────────
    if cov_type == "cluster":
        if cluster is None:
            raise ValueError("cluster= must be provided when cov_type='cluster'.")
        clusters = [cluster] if isinstance(cluster, str) else list(cluster)
        for c in clusters:
            if c not in data.columns:
                raise errors.cluster_column_error(c, data.columns.tolist())
    elif cluster is not None:
        raise ValueError("cluster= is only used when cov_type='cluster'.")
    if cov_type == "HAC" and max_lags is None:
        raise ValueError("max_lags= must be provided when cov_type='HAC'.")
    if time is not None and time not in data.columns:
        raise errors.missing_column_error(time, data.columns.tolist())

    if not isinstance(start_values, dict) or len(start_values) == 0:
        raise ValueError("start_values must be a non-empty dict of param_name -> value.")

    # ── formula parsing ─────────────────────────────────────────────
    if "~" not in formula:
        raise ValueError("formula must be two-sided, e.g. 'y ~ a * exp(-b * x) + c'.")
    lhs, rhs = formula.split("~", 1)
    y_name = lhs.strip()
    rhs = rhs.strip()
    if not y_name:
        raise ValueError("formula LHS (dependent variable) is empty.")
    if y_name not in data.columns:
        raise errors.missing_column_error(y_name, data.columns.tolist())

    try:
        expr = parse_expr(
            rhs,
            transformations=standard_transformations + (convert_xor,),
            evaluate=True,
        )
    except Exception as e:  # pragma: no cover - defensive
        raise ValueError(f"Could not parse RHS expression {rhs!r}: {e}") from e

    # Classify every free symbol as parameter / data / error.
    sym_by_name = {str(s): s for s in expr.free_symbols}
    param_names: list[str] = list(start_values.keys())
    for name in param_names:
        if name not in sym_by_name:
            raise ValueError(
                f"Parameter '{name}' in start_values does not appear in the RHS "
                f"expression {rhs!r}."
            )
    data_names: list[str] = []
    for name, s in sym_by_name.items():
        is_param = name in start_values
        is_data = name in data.columns
        if is_param and is_data:
            raise ValueError(
                f"Symbol '{name}' is both a parameter (in start_values) and a "
                f"data column. Rename one to disambiguate -- the estimator will "
                f"not guess which you meant."
            )
        if is_param:
            continue
        if is_data:
            data_names.append(name)
        else:
            raise ValueError(
                f"Symbol '{name}' is neither a key in start_values nor a column "
                f"in data -- likely a typo. RHS symbols must be parameters "
                f"(in start_values) or data columns."
            )

    param_syms = [sym_by_name[n] for n in param_names]
    data_syms = [sym_by_name[n] for n in data_names]

    # Extract and align arrays; drop rows with any missing value.
    y_arr = data[y_name].to_numpy(dtype=float)
    data_arrays = [data[n].to_numpy(dtype=float) for n in data_names]
    keep = ~np.isnan(y_arr)
    for a in data_arrays:
        keep &= ~np.isnan(a)
    if not keep.all():
        y_arr = y_arr[keep]
        data_arrays = [a[keep] for a in data_arrays]
    n_obs = int(len(y_arr))
    k = len(param_names)
    if n_obs <= k:
        raise ValueError(
            f"Not enough non-missing observations ({n_obs}) to estimate "
            f"{k} parameters."
        )
    x0 = np.asarray([float(start_values[n]) for n in param_names], dtype=float)

    # Mean function: f(*params, *data_cols) -> fitted values.
    mean_func = _sym_lambdify([*param_syms, *data_syms], expr, modules=["numpy"])

    def resid(beta: np.ndarray) -> np.ndarray:
        args = [*np.asarray(beta, dtype=float), *data_arrays]
        fitted = np.asarray(mean_func(*args), dtype=float).ravel()
        return y_arr - fitted

    # Analytic Jacobian of the mean function; fall back to numerical if any
    # parameter fails to differentiate / lambdify.
    jac_func: Callable[..., np.ndarray] | None = None
    jacobian_method = "analytic"
    jac_fallback_reason = ""
    try:
        d_exprs = [_sym_diff(expr, s) for s in param_syms]
        d_funcs = [
            _sym_lambdify([*param_syms, *data_syms], d, modules=["numpy"])
            for d in d_exprs
        ]

        def jac_func(beta: np.ndarray) -> np.ndarray:
            args = [*np.asarray(beta, dtype=float), *data_arrays]
            n = len(y_arr)
            cols = []
            for g in d_funcs:
                col = np.asarray(g(*args), dtype=float).ravel()
                if col.size == 1:
                    # Constant parameter derivative (e.g. d/dc of "... + c")
                    # lambdifies to a scalar; broadcast it to length n.
                    col = np.full(n, col.item())
                cols.append(col)
            J_mean = np.column_stack(cols)  # (n, k): d f / d theta
            # least_squares needs d(residual)/d(theta) = - d f / d theta.
            return -J_mean

        _ = jac_func(x0)  # smoke test
    except Exception as e:  # pragma: no cover - defensive
        jac_func = None
        jacobian_method = "numerical"
        jac_fallback_reason = str(e)

    # ── solve ───────────────────────────────────────────────────────
    fit_kwargs = dict(solver_kwargs)
    if "jac" in fit_kwargs:
        raise ValueError(
            "Pass jac via the analytic differentiator; do not set jac= in "
            "solver_kwargs."
        )
    fit_kwargs["jac"] = jac_func if jac_func is not None else "2-point"
    res = least_squares(resid, x0, **fit_kwargs)

    beta = np.asarray(res.x, dtype=float)
    resid_sol = np.asarray(res.fun, dtype=float).ravel()

    # Jacobian at the solution (d f / d theta), shape (n, k).
    if jac_func is not None:
        J_mean = -jac_func(beta)
    else:
        # res.jac = d(residual)/d(theta) = - d f / d theta.
        J_mean = -np.asarray(res.jac, dtype=float)

    # ── covariance ──────────────────────────────────────────────────
    cov_label = cov_type
    V: np.ndarray
    try:
        if cov_type in ("HC0", "HC1", "HC2", "HC3"):
            V = white_cov(J_mean, resid_sol, kind=cov_type)
            cov_label = cov_type
        elif cov_type == "nonrobust":
            ssr = float(np.sum(resid_sol ** 2))
            sigma2 = ssr / (n_obs - k)
            JtJ_inv = np.linalg.inv(J_mean.T @ J_mean)
            V = sigma2 * JtJ_inv
            cov_label = "nonrobust"
        elif cov_type == "cluster":
            from open_econs.core.cov import _as_int_labels, multiway_cluster_cov

            groups = [
                _as_int_labels(data[c].to_numpy(dtype=float)[keep])
                for c in clusters
            ]
            V = multiway_cluster_cov(J_mean, resid_sol, groups)
            cov_label = "cluster(" + ", ".join(clusters) + ")"
        else:  # HAC
            from open_econs.core.cov import _as_int_labels, newey_west_cov

            # HAC orders observations by ``time`` when supplied (otherwise the
            # current row order), mirroring oe.ols(..., cov_type="HAC",
            # time=...).  A cluster id (when ``cluster`` is a string) aggregates
            # scores across entities first, matching Stata's panel HAC.
            cl = (
                _as_int_labels(data[cluster].to_numpy(dtype=float)[keep])
                if isinstance(cluster, str)
                else None
            )
            time_index = (
                data[time].to_numpy(dtype=float)[keep] if time is not None else None
            )
            assert max_lags is not None  # guaranteed by the validation above
            V = newey_west_cov(
                J_mean, resid_sol, max_lags=max_lags, time_index=time_index,
                cluster=cl, adjust=False,
            )
            cov_label = f"HAC({max_lags})" + (
                f" cluster({cluster})" if isinstance(cluster, str) else ""
            )
    except Exception as e:
        import warnings as _w

        _w.warn(
            f"Covariance computation failed ({e}); standard errors set to NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        V = np.full((k, k), np.nan)

    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    t_stats = np.where(se > 0, beta / se, np.nan)
    p_values = 2.0 * (1.0 - _norm.cdf(np.abs(t_stats)))
    conf_int = pd.DataFrame(
        {"lower": beta - 1.96 * se, "upper": beta + 1.96 * se},
        index=param_names,
    )

    coefficients = pd.Series(beta, index=param_names, name="coefficient")
    std_errors = pd.Series(se, index=param_names, name="std_error")

    fitted_values = pd.Series(
        y_arr - resid_sol, index=data.index[keep], name="fitted"
    )
    residuals = pd.Series(resid_sol, index=data.index[keep], name="residuals")

    call = _capture_call(
        formula=formula,
        start_values=start_values,
        cov_type=cov_type,
        cluster=cluster,
        max_lags=max_lags,
        time=time,
        jacobian_method=jacobian_method,
        **solver_kwargs,
    )
    if jac_fallback_reason:
        call["jacobian_fallback_reason"] = jac_fallback_reason

    return NLSResult(
        formula=formula,
        rhs_formula=rhs,
        coefficients=coefficients,
        std_errors=std_errors,
        t_stats=pd.Series(t_stats, index=param_names, name="t"),
        p_values=pd.Series(p_values, index=param_names, name="p_value"),
        conf_int=conf_int,
        cov_type=cov_label,
        n_obs=n_obs,
        jacobian_method=jacobian_method,
        success=res.success,
        n_function_evaluations=res.nfev,
        n_jacobian_evaluations=res.njev,
        cost=float(res.cost),
        status=res.status,
        message=res.message,
        optimality=float(res.optimality),
        vcov_matrix=V,
        fitted_values=fitted_values,
        residuals=residuals,
        call=call,
    )
