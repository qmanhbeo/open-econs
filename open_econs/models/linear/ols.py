from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from open_econs._version import __version__
from open_econs._internal import errors
from open_econs._internal.formula import parse_formula
from open_econs.core.results import OLSResult


def ols(
    formula: str,
    data: pd.DataFrame,
    cluster: str | None = None,
    cov_type: str = "HC1",
) -> OLSResult:
    """Estimate an ordinary least-squares regression.

    Parameters
    ----------
    formula : str
        Two-sided formula string, e.g. ``"income ~ education + age"``.
    data : pd.DataFrame
        Data containing all variables referenced in *formula*.
    cluster : str, optional
        Column name for cluster-robust standard errors.
    cov_type : str, default "HC1"
        Covariance estimator type. Common choices: ``"HC1"`` (default,
        matches Stata ``reg, robust``), ``"HC0"``, ``"HC2"``, ``"HC3"``,
        ``"nonrobust"``. Ignored when *cluster* is provided (cluster-robust
        is used instead).

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
    call = _capture_call(formula=formula, cluster=cluster, cov_type=cov_type)
    yy, XX = parse_formula(formula, data)

    y_arr = yy.values.ravel()

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

    return OLSResult(
        formula=formula,
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
        f_statistic=_safe_fvalue(fitted),
        f_p_value=_safe_f_pvalue(fitted),
        rsd=float(np.sqrt(fitted.mse_resid)),
        fitted=fitted_values,
        residuals=residuals,
        call=call,
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