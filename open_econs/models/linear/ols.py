from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from open_econs._version import __version__
from open_econs._internal import errors
from open_econs.core.results import OLSResult


def ols(
    formula: str,
    data: pd.DataFrame,
    cluster: str | None = None,
    cov_type: str = "HC2",
    weights: str | np.ndarray | pd.Series | None = None,
) -> OLSResult:
    """Estimate an ordinary least-squares or weighted least-squares regression.

    Parameters
    ----------
    formula : str
        Two-sided formula string, e.g. ``"income ~ education + age"``.
    data : pd.DataFrame
        Data containing all variables referenced in *formula*.
    cluster : str, optional
        Column name for cluster-robust standard errors.
    cov_type : str, default "HC2"
        Covariance estimator type. Common choices: ``"HC2"`` (default,
        matches modern Stata ``reg, robust``; changed from HC1 in v0.2.0),
        ``"HC1`` (classic White SE), ``"HC0"``, ``"HC3"``, ``"nonrobust"``.
        Ignored when *cluster* is provided (cluster-robust is used instead).
    weights : str or array-like, optional
        Frequency/analytic weights for weighted least squares. If a string,
        interpreted as a column name in *data*. If an array, must match the
        number of rows in *data*.

    Returns
    -------
    OLSResult
        Immutable result object with named coefficient arrays.

    Examples
    --------
    >>> import open_econs as oe
    >>> result = oe.ols("income ~ education + age", data=df)
    >>> result.tidy()
    >>> result.coefficients["education"]
    """
    call = _capture_call(formula=formula, cluster=cluster, cov_type=cov_type, weights=weights)
    rhs_formula = formula.split("~", 1)[1].strip()

    from formulaic import Formula
    try:
        formula_obj = Formula(formula)
        model_spec = formula_obj.get_model_matrix(data, na_action="drop")
    except Exception as e:
        msg = str(e)
        if "not present in the dataset" in msg or "is not present" in msg:
            import re as _re
            m = _re.search(r"`(\w+)`", msg)
            bad_col = m.group(1) if m else formula.split("~", 1)[1].strip()
            raise errors.missing_column_error(bad_col, data.columns.tolist()) from e
        raise
    if hasattr(model_spec, "rhs"):
        XX = model_spec.rhs
        yy = model_spec.lhs
    else:
        from open_econs._internal.formula import parse_formula as _parse
        yy, XX = _parse(formula, data)
        model_spec = None

    original_n = len(data)
    dropped = original_n - len(yy)
    vars_needed = {str(v) for v in formula_obj.required_variables}
    cols_with_nas = sorted(v for v in vars_needed if v in data.columns and data[v].isna().any())
    if dropped > 0:
        from open_econs._internal.errors import rows_dropped_warning
        import warnings as _w
        _w.warn(rows_dropped_warning(dropped, original_n, cols_with_nas), RuntimeWarning, stacklevel=3)

    if len(yy) == 0:
        from open_econs._internal.errors import empty_data_error
        raise empty_data_error(original_n, dropped, cols_with_nas)

    y_arr = yy.values.ravel()

    if model_spec is not None:
        stored_spec = model_spec.model_spec.rhs
    else:
        stored_spec = None

    condition_number = _check_collinearity(XX)

    if weights is not None:
        if isinstance(weights, str):
            if weights not in data.columns:
                raise errors.missing_column_error(weights, data.columns.tolist())
            w_arr = data.loc[XX.index, weights].values.astype(float)
        else:
            w_arr = np.asarray(weights, dtype=float)
            if len(w_arr) != original_n:
                raise ValueError(
                    f"weights array length ({len(w_arr)}) does not match data rows ({original_n})"
                )
            w_arr = w_arr[XX.index]
        if np.any(w_arr < 0):
            raise ValueError("Weights must be non-negative.")
        fitted = sm.WLS(y_arr, XX.values, weights=w_arr).fit(
            cov_type="cluster" if cluster else cov_type,
            cov_kwds={"groups": data.loc[XX.index, cluster]} if cluster else {},
        )
        cov_label = f"cluster({cluster})" if cluster else cov_type
    else:
        if cluster is not None:
            if cluster not in data.columns:
                raise errors.cluster_column_error(cluster, data.columns.tolist())
            aligned_groups = data.loc[XX.index, cluster]
            fitted = sm.OLS(y_arr, XX.values).fit(
                cov_type="cluster",
                cov_kwds={"groups": aligned_groups},
            )
            cov_label = f"cluster({cluster})"
        else:
            fitted = sm.OLS(y_arr, XX.values).fit(cov_type=cov_type)
            cov_label = cov_type

    coef_arr = fitted.params
    se_arr = fitted.bse
    t_arr = fitted.tvalues
    p_arr = fitted.pvalues
    conf_arr = fitted.conf_int()

    conf_int = pd.DataFrame(
        {"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]},
        index=XX.columns,
    )

    fitted_values = pd.Series(fitted.fittedvalues, index=XX.index, name="fitted")
    residuals = pd.Series(fitted.resid, index=XX.index, name="residuals")

    f_stat = _safe_fvalue(fitted)
    f_pval = _safe_f_pvalue(fitted)

    return OLSResult(
        formula=formula,
        rhs_formula=rhs_formula,
        nobs=int(fitted.nobs),
        df_resid=int(fitted.df_resid),
        df_model=int(fitted.df_model),
        cov_type=cov_label,
        coefficients=pd.Series(coef_arr, index=XX.columns),
        std_errors=pd.Series(se_arr, index=XX.columns),
        t_stats=pd.Series(t_arr, index=XX.columns),
        p_values=pd.Series(p_arr, index=XX.columns),
        conf_int=conf_int,
        r_squared=float(fitted.rsquared),
        adj_r_squared=float(fitted.rsquared_adj),
        f_statistic=f_stat,
        f_p_value=f_pval,
        rsd=float(np.sqrt(fitted.mse_resid)),
        llf=_safe_llf(fitted),
        aic=_safe_aic(fitted),
        bic=_safe_bic(fitted),
        fitted=fitted_values,
        residuals=residuals,
        call=call,
        model_spec=stored_spec,
        condition_number=condition_number,
        _X=XX,
        _sm_fit=fitted,
    )


reg = ols


def _capture_call(**kwargs: Any) -> dict[str, Any]:
    kwargs["timestamp"] = str(datetime.now())
    kwargs["package_version"] = __version__
    return kwargs


def _safe_fvalue(fitted: Any) -> float:
    try:
        return float(fitted.fvalue)
    except (ValueError, AttributeError):
        return float("nan")


def _safe_f_pvalue(fitted: Any) -> float:
    try:
        return float(fitted.f_pvalue)
    except (ValueError, AttributeError):
        return float("nan")


def _safe_llf(fitted: Any) -> float:
    try:
        return float(fitted.llf)
    except (ValueError, AttributeError):
        return float("nan")


def _safe_aic(fitted: Any) -> float:
    try:
        return float(fitted.aic)
    except (ValueError, AttributeError):
        return float("nan")


def _safe_bic(fitted: Any) -> float:
    try:
        return float(fitted.bic)
    except (ValueError, AttributeError):
        return float("nan")


def _check_collinearity(XX: pd.DataFrame) -> float:
    from numpy.linalg import cond, matrix_rank
    X_vals = XX.values
    n_params = X_vals.shape[1]
    rank = matrix_rank(X_vals)
    if rank < n_params:
        raise errors.singular_matrix_error()
    X_no_intercept = X_vals[:, [c != "Intercept" for c in XX.columns]]
    if X_no_intercept.shape[1] == 0:
        return float(cond(X_vals))
    from numpy import std as _std
    X_scaled = X_no_intercept / _std(X_no_intercept, axis=0)
    cn = float(cond(X_scaled))
    if cn > 30:
        import warnings as _w
        _w.warn(
            f"Design matrix is near-singular (condition number = {cn:.2e}). "
            "Belsley, Kuh & Welsch (1980) recommend caution above 30. "
            "Consider removing collinear predictors.",
            RuntimeWarning,
            stacklevel=3,
        )
    return cn