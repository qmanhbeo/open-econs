
import pandas as pd
import statsmodels.api as sm

from open_econs.core.call_capture import capture_call as _capture_call
from open_econs._internal import errors
from open_econs.core.results import BinaryResult


def logit(
    formula: str,
    data: pd.DataFrame,
    cov_type: str = "nonrobust",
) -> BinaryResult:
    """Estimate a binary logit model.

    Parameters
    ----------
    formula : str
        Two-sided formula string, e.g. ``"y ~ x1 + x2"``.
    data : pd.DataFrame
        Data containing all variables referenced in *formula*.
    cov_type : str, default "nonrobust"
        Covariance estimator type. Use ``"HC0"`` for robust SEs.

    Returns
    -------
    BinaryResult
        Immutable result object with coefficient arrays and margins.

    Examples
    --------
    >>> import open_econs as oe
    >>> r = oe.logit("y ~ x1 + x2", data=df)
    >>> r.tidy()
    >>> r.margins()
    """
    call = _capture_call(formula=formula, cov_type=cov_type, model_type="logit")
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

    original_n = len(data)
    dropped = original_n - len(yy)
    if dropped > 0:
        import warnings as _w
        _w.warn(
            errors.rows_dropped_warning(dropped, original_n, []),
            RuntimeWarning,
            stacklevel=3,
        )

    if len(yy) == 0:
        raise errors.empty_data_error(original_n, dropped, [])

    y_arr = yy.values.ravel()
    X_arr = XX.values

    _check_collinearity(XX)

    fitted = sm.Logit(y_arr, X_arr).fit(disp=False, cov_type=cov_type)

    coef_arr = fitted.params
    se_arr = fitted.bse
    z_arr = fitted.tvalues
    p_arr = fitted.pvalues
    conf_arr = fitted.conf_int()

    conf_int = pd.DataFrame(
        {"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]},
        index=XX.columns,
    )

    fitted_values = pd.Series(fitted.predict(X_arr), index=XX.index, name="predicted_proba")

    return BinaryResult(
        formula=formula,
        rhs_formula=rhs_formula,
        nobs=int(fitted.nobs),
        df_resid=int(fitted.df_resid),
        df_model=int(fitted.df_model),
        cov_type=cov_type,
        coefficients=pd.Series(coef_arr, index=XX.columns),
        std_errors=pd.Series(se_arr, index=XX.columns),
        z_stats=pd.Series(z_arr, index=XX.columns),
        p_values=pd.Series(p_arr, index=XX.columns),
        conf_int=conf_int,
        llf=float(fitted.llf),
        aic=float(fitted.aic),
        bic=float(fitted.bic),
        pseudo_r2=float(fitted.prsquared),
        fitted=fitted_values,
        call=call,
        model_type="logit",
        _fit=fitted,
    )


def _check_collinearity(XX: pd.DataFrame) -> None:
    from numpy.linalg import matrix_rank
    X_vals = XX.values
    if matrix_rank(X_vals) < X_vals.shape[1]:
        raise errors.singular_matrix_error()