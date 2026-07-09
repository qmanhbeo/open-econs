from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from open_econs._version import __version__
from open_econs._internal import errors
from open_econs.core.results import OLSResult


def fe(
    formula: str,
    data: pd.DataFrame,
    entity: str | None = None,
    time: str | None = None,
    cluster: str | None = None,
    cov_type: str = "HC1",
) -> OLSResult:
    """Estimate a linear fixed-effects (within) model.

    Parameters
    ----------
    formula : str
        Two-sided formula string, e.g. ``"y ~ x1 + x2"``. Do *not* include
        the fixed-effect indicator in the formula; use *entity* and *time*.
    data : pd.DataFrame
        Data containing all variables referenced in *formula* plus the
        entity/time columns.
    entity : str, optional
        Column name for entity (panel unit) fixed effects. If provided, both
        *y* and *X* are group-demeaned within each entity.
    time : str, optional
        Column name for time fixed effects. If *entity* is also provided,
        two-way fixed effects are used (entity and time dummies absorbed
        via group-demeaning).
    cluster : str, optional
        Column name for cluster-robust standard errors.
    cov_type : str, default "HC1"
        Covariance estimator type. Used when *cluster* is not set.

    Returns
    -------
    OLSResult
        OLS result from the within-transformed regression. Standard errors
        are adjusted for the degrees of freedom lost to absorbed FE.

    Examples
    --------
    >>> import open_econs as oe
    >>> r = oe.fe("y ~ x1 + x2", data=df, entity="country", time="year")
    >>> r.tidy()
    """
    call = _capture_call(
        formula=formula, entity=entity, time=time, cluster=cluster, cov_type=cov_type,
    )

    if entity is None and time is None:
        raise ValueError("At least one of entity= or time= must be provided.")

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
        XX = model_spec.rhs.copy()
        yy = model_spec.lhs.copy()
    else:
        from open_econs._internal.formula import parse_formula as _parse
        yy_df, XX_df = _parse(formula, data)
        XX = XX_df.copy()
        yy = yy_df.copy()

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

    y_arr = yy.values.ravel().astype(float)
    X_arr = XX.values.astype(float)
    entity_arr = data.loc[XX.index, entity].values if entity else None
    time_arr = data.loc[XX.index, time].values if time else None

    n_absorbed = 0

    if entity is not None:
        entity_dummies = pd.get_dummies(entity_arr)
        n_absorbed += entity_dummies.shape[1]
        y_arr = _demean(y_arr, entity_dummies.values)
        X_arr = _demean(X_arr, entity_dummies.values)

    if time is not None and entity is not None:
        time_dummies = pd.get_dummies(time_arr)
        already_absorbed = set()
        for col in time_dummies.columns:
            if col in entity_dummies.columns:
                already_absorbed.add(col)
        to_demean = [c for c in time_dummies.columns if c not in already_absorbed]
        if to_demean:
            td = time_dummies[to_demean].values
            n_absorbed += td.shape[1]
            y_arr = _demean(y_arr, td)
            X_arr = _demean(X_arr, td)
    elif time is not None:
        time_dummies = pd.get_dummies(time_arr)
        n_absorbed += time_dummies.shape[1]
        y_arr = _demean(y_arr, time_dummies.values)
        X_arr = _demean(X_arr, time_dummies.values)

    import statsmodels.api as sm

    if cluster is not None:
        if cluster not in data.columns:
            raise errors.cluster_column_error(cluster, data.columns.tolist())
        aligned_groups = data.loc[XX.index, cluster]
        fitted = sm.OLS(y_arr, X_arr).fit(
            cov_type="cluster",
            cov_kwds={"groups": aligned_groups},
        )
        cov_label = f"cluster({cluster})"
    else:
        fitted = sm.OLS(y_arr, X_arr).fit(cov_type=cov_type)
        cov_label = cov_type

    coef_arr = fitted.params
    se_arr = fitted.bse
    t_arr = fitted.tvalues
    p_arr = fitted.pvalues
    conf_arr = fitted.conf_int()

    coef_arr = fitted.params
    se_arr = fitted.bse
    t_arr = fitted.tvalues
    p_arr = fitted.pvalues
    conf_arr = fitted.conf_int()

    kept_columns = [c for c in XX.columns if c != "Intercept"]

    n_real = len(kept_columns)
    coef_arr = coef_arr[-n_real:] if len(coef_arr) > n_real else coef_arr
    se_arr = se_arr[-n_real:] if len(se_arr) > n_real else se_arr
    t_arr = t_arr[-n_real:] if len(t_arr) > n_real else t_arr
    p_arr = p_arr[-n_real:] if len(p_arr) > n_real else p_arr
    if conf_arr.shape[0] > n_real:
        conf_arr = conf_arr[-n_real:]

    conf_int = pd.DataFrame(
        {"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]},
        index=kept_columns,
    )

    nobs = int(fitted.nobs)
    df_resid = int(fitted.df_resid)
    df_model = int(fitted.df_model)

    fitted_values = pd.Series(fitted.fittedvalues, index=XX.index, name="fitted")
    residuals = pd.Series(fitted.resid, index=XX.index, name="residuals")

    rhs_formula = formula.split("~", 1)[1].strip()

    return OLSResult(
        formula=formula,
        rhs_formula=rhs_formula,
        nobs=nobs,
        df_resid=df_resid,
        df_model=df_model,
        cov_type=cov_label,
        coefficients=pd.Series(coef_arr, index=kept_columns),
        std_errors=pd.Series(se_arr, index=kept_columns),
        t_stats=pd.Series(t_arr, index=kept_columns),
        p_values=pd.Series(p_arr, index=kept_columns),
        conf_int=conf_int,
        r_squared=float(fitted.rsquared),
        adj_r_squared=float(fitted.rsquared_adj),
        f_statistic=_safe_fvalue(fitted),
        f_p_value=_safe_f_pvalue(fitted),
        rsd=float(np.sqrt(fitted.mse_resid)),
        llf=_safe_llf(fitted),
        aic=_safe_aic(fitted),
        bic=_safe_bic(fitted),
        fitted=fitted_values,
        residuals=residuals,
        call=call,
        condition_number=float(np.linalg.cond(X_arr)) if X_arr.shape[1] > 0 else 0.0,
        _X=XX,
        _sm_fit=fitted,
    )


def _demean(y: np.ndarray, dummies: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    centered = y - dummies @ np.linalg.lstsq(dummies, y, rcond=None)[0]
    return centered.ravel() if y.shape[1] == 1 else centered


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