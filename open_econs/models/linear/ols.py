from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from open_econs.core.call_capture import capture_call as _capture_call
from open_econs._internal import errors
from open_econs.core.results import OLSResult
from open_econs.core.cov_type import validate_cov_type


def ols(
    formula: str,
    data: pd.DataFrame,
    cluster: str | list[str] | None = None,
    cov_type: str = "HC2",
    weights: str | np.ndarray | pd.Series | None = None,
    lags: int | None = None,
    time: str | None = None,
    hac_adjust: bool = False,
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

        Set ``cov_type="HAC"`` to use Newey-West (1987) heteroskedasticity- and
        autocorrelation-robust standard errors; the number of lags is given by
        *lags* and the time ordering by *time* (or *cluster* for panel HAC).
    cluster : str or list of str, optional
        Column name(s) for cluster-robust standard errors.  Passing a *list*
        requests multi-way clustering (e.g. ``["firm", "year"]``), implemented
        via the Cameron-Gelbach-Miller (2011) minik estimator.  Ignored when
        ``cov_type="HAC"``.
    lags : int, optional
        Number of lags for Newey-West HAC (required when ``cov_type="HAC"``).
    time : str, optional
        Column with the time index used to order observations for Newey-West
        HAC (or the panel time id when combined with *cluster*).
    hac_adjust : bool, default False
        Degrees-of-freedom correction for Newey-West HAC standard errors.

        When ``True``, the HAC variance is multiplied by ``N / (N - K)``
        (where N = observations, K = number of parameters including
        intercept).  This is the N/(N-K) adjustment borrowed from White's
        HC1 (MacKinnon & White, 1985) and applied unconditionally by
        Stata's ``newey`` command.  The original Newey & West (1987) paper
        does **not** include this correction — it is a finite-sample ad-hoc
        adjustment with no theoretical HAC-specific justification.

        **Implementation comparison:**
        ================================ =================== ==============
        Implementation                    Applies N/(N-K)?    Default
        ================================ =================== ==============
        Newey & West (1987)               No                  —
        **Open-econs** (current)          **No**              **``False``**
        Statsmodels ``fit(cov_type=...)`` No                  Default
        R ``sandwich::NeweyWest()``       No                  ``adjust=FALSE``
        Stata ``newey``                   Yes                 Always (no opt-out)
        MATLAB ``hac``                    Yes                 Default
        ================================ =================== ==============

        Set ``hac_adjust=True`` for SEs that match Stata.  Leave ``False``
        (default) for the original NW1987 formula, matching statsmodels and
        R's sandwich defaults.

        References
        ----------
        - Newey, W. K. & West, K. D. (1987). "A Simple, Positive Semi-Definite,
          Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
          *Econometrica*, 55(3), 703–708.
        - MacKinnon, J. G. & White, H. (1985). "Some Heteroskedasticity-Consistent
          Covariance Matrix Estimators with Improved Finite Sample Properties."
          *Journal of Econometrics*, 29(3), 305–325.
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
    cov_type = validate_cov_type(
        cov_type,
        accepted={"nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"},
        estimator="ols()",
    )

    call = _capture_call(
        formula=formula, cluster=cluster, cov_type=cov_type, weights=weights,
        lags=lags, time=time, hac_adjust=hac_adjust,
    )
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

    clusters = list(cluster) if isinstance(cluster, (list, tuple)) else None
    multiway = clusters is not None and len(clusters) >= 1
    use_hac = cov_type == "HAC"

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
        if multiway or use_hac:
            raise ValueError(
                "Weights are not supported together with multi-way clustering "
                "or Newey-West HAC in this version."
            )
        fitted = sm.WLS(y_arr, XX.values, weights=w_arr).fit(
            cov_type="cluster" if isinstance(cluster, str) else cov_type,
            cov_kwds={"groups": data.loc[XX.index, cluster]} if isinstance(cluster, str) else {},
        )
        cov_label = f"cluster({cluster})" if isinstance(cluster, str) else cov_type
        coef_arr = fitted.params
        se_arr = fitted.bse
        t_arr = fitted.tvalues
        p_arr = fitted.pvalues
        conf_arr = fitted.conf_int()
    else:
        # Fit OLS (nonrobust) to obtain point estimates and residuals; the
        # covariance is then computed explicitly below so we can support
        # multi-way clustering and Newey-West HAC.
        fitted = sm.OLS(y_arr, XX.values).fit(cov_type="nonrobust")
        coef_arr = fitted.params
        if multiway:
            assert clusters is not None  # guaranteed by `multiway`
            from open_econs.core.cov import multiway_cluster_cov, _as_int_labels

            groups = [_as_int_labels(data.loc[XX.index, c].values) for c in clusters]
            V = multiway_cluster_cov(XX.values, fitted.resid, groups)
            se_arr = np.sqrt(np.maximum(np.diag(V), 0.0))
            cov_label = "cluster(" + ", ".join(clusters) + ")"
            t_arr = np.where(se_arr > 0, coef_arr / se_arr, np.nan)
            from scipy.stats import norm as _norm

            p_arr = 2.0 * (1.0 - _norm.cdf(np.abs(t_arr)))
            conf_arr = np.column_stack(
                [coef_arr - 1.96 * se_arr, coef_arr + 1.96 * se_arr]
            )
        elif use_hac:
            from open_econs.core.cov import newey_west_cov, _as_int_labels

            if lags is None:
                raise ValueError("Newey-West HAC requires `lags` (e.g. lags=1).")
            time_arr = data.loc[XX.index, time].values if time else None
            cl = _as_int_labels(data.loc[XX.index, cluster].values) if isinstance(cluster, str) else None
            V = newey_west_cov(
                XX.values, fitted.resid, max_lags=lags,
                time_index=time_arr, cluster=cl,
                adjust=hac_adjust,
            )
            se_arr = np.sqrt(np.maximum(np.diag(V), 0.0))
            cov_label = f"HAC({lags})" + (f" cluster({cluster})" if isinstance(cluster, str) else "")
            t_arr = np.where(se_arr > 0, coef_arr / se_arr, np.nan)
            from scipy.stats import norm as _norm

            p_arr = 2.0 * (1.0 - _norm.cdf(np.abs(t_arr)))
            conf_arr = np.column_stack(
                [coef_arr - 1.96 * se_arr, coef_arr + 1.96 * se_arr]
            )
        elif isinstance(cluster, str):
            if cluster not in data.columns:
                raise errors.cluster_column_error(cluster, data.columns.tolist())
            fitted = sm.OLS(y_arr, XX.values).fit(
                cov_type="cluster",
                cov_kwds={"groups": data.loc[XX.index, cluster]},
            )
            se_arr = fitted.bse
            t_arr = fitted.tvalues
            p_arr = fitted.pvalues
            conf_arr = fitted.conf_int()
            cov_label = f"cluster({cluster})"
        else:
            fitted = sm.OLS(y_arr, XX.values).fit(cov_type=cov_type)
            se_arr = fitted.bse
            t_arr = fitted.tvalues
            p_arr = fitted.pvalues
            conf_arr = fitted.conf_int()
            cov_label = cov_type

    conf_int = pd.DataFrame(
        {"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]},
        index=XX.columns,
    )

    fitted_values = pd.Series(fitted.fittedvalues, index=XX.index, name="fitted")
    residuals = pd.Series(fitted.resid, index=XX.index, name="residuals")

    f_stat = _safe_fvalue(fitted)
    f_pval = _safe_f_pvalue(fitted)

    result = OLSResult(
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
        _fit=fitted,
    )
    if "V" in dir() and (multiway or use_hac):
        object.__setattr__(
            result,
            "_cov",
            pd.DataFrame(V, index=XX.columns, columns=XX.columns),
        )
    return result


reg = ols


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