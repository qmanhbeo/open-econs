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
        via iterative demeaning for unbalanced panels).
    cluster : str, optional
        Column name for cluster-robust standard errors.
    cov_type : str, default "HC1"
        Covariance estimator type. Used when *cluster* is not set.

    Returns
    -------
    OLSResult
        OLS result from the within-transformed regression. Degrees of
        freedom are adjusted for absorbed FE, and for two-way panels
        the iterative (alternating-projections) demeaning is used so
        unbalanced panels produce correct estimates.

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

    n_absorbed = 0

    if entity is not None and time is not None:
        entity_arr = data.loc[XX.index, entity].values
        time_arr = data.loc[XX.index, time].values
        unique_entities = np.unique(entity_arr)
        unique_times = np.unique(time_arr)
        n_entity = len(unique_entities)
        n_time = len(unique_times)
        n_absorbed = n_entity + n_time - 1
        y_arr, X_arr = _demean_two_way(y_arr, X_arr, entity_arr, time_arr)
    elif entity is not None:
        entity_arr = data.loc[XX.index, entity].values
        unique_entities = np.unique(entity_arr)
        n_absorbed = len(unique_entities)
        y_arr = _demean(y_arr, entity_arr)
        X_arr = _demean(X_arr, entity_arr)
    else:
        time_arr = data.loc[XX.index, time].values
        unique_times = np.unique(time_arr)
        n_absorbed = len(unique_times)
        y_arr = _demean(y_arr, time_arr)
        X_arr = _demean(X_arr, time_arr)

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

    k = X_arr.shape[1]
    n = int(fitted.nobs)
    df_resid_adj = n - k - n_absorbed
    df_model_adj = k

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

    r2 = 1.0 - np.sum(fitted.resid ** 2) / (np.sum((y_arr - np.mean(y_arr)) ** 2) + 1e-15)
    if np.isnan(r2) or r2 < 0 or r2 > 1:
        r2 = float(fitted.rsquared)

    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(df_resid_adj, 1)

    fitted_values = pd.Series(fitted.fittedvalues, index=XX.index, name="fitted")
    residuals = pd.Series(fitted.resid, index=XX.index, name="residuals")

    rhs_formula = formula.split("~", 1)[1].strip()

    return OLSResult(
        formula=formula,
        rhs_formula=rhs_formula,
        nobs=n,
        df_resid=df_resid_adj,
        df_model=df_model_adj,
        cov_type=cov_label,
        coefficients=pd.Series(coef_arr, index=kept_columns),
        std_errors=pd.Series(se_arr, index=kept_columns),
        t_stats=pd.Series(t_arr, index=kept_columns),
        p_values=pd.Series(p_arr, index=kept_columns),
        conf_int=conf_int,
        r_squared=float(r2),
        adj_r_squared=float(adj_r2),
        f_statistic=_safe_fvalue(fitted),
        f_p_value=_safe_f_pvalue(fitted),
        rsd=float(np.sqrt(np.sum(fitted.resid ** 2) / max(df_resid_adj, 1))),
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


def _demean(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    dummies = pd.get_dummies(pd.Series(groups)).values
    resid = y - dummies @ np.linalg.lstsq(dummies, y, rcond=None)[0]
    return resid.ravel() if y.shape[1] == 1 else resid


def _demean_two_way(
    y: np.ndarray | pd.Series,
    x: np.ndarray | pd.DataFrame,
    entities: np.ndarray,
    times: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative (alternating-projections) demeaning for two-way FE.

    This computes the within transformation y - y_bar_i - y_bar_t + y_bar
    without explicitly constructing the dummy matrix, using the algorithm
    from the Stata reghdfe package (Correia 2017). This is exact for
    unbalanced panels.

    Returns (y_demeaned, X_demeaned) as numpy arrays.
    """
    if isinstance(y, pd.Series):
        y_arr = y.values.ravel().astype(float)
    else:
        y_arr = y.ravel().astype(float) if y.ndim <= 2 else y.astype(float)

    if isinstance(x, pd.DataFrame):
        X_arr = x.values.astype(float)
    else:
        X_arr = x.astype(float)

    if x.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    for iteration in range(max_iter):
        prev_y = y_arr.copy()
        prev_X = X_arr.copy()

        y_arr = _within_transform(y_arr, entities)
        X_arr = _within_transform(X_arr, entities)
        y_arr = _within_transform(y_arr, times)
        X_arr = _within_transform(X_arr, times)

        y_change = np.max(np.abs(y_arr - prev_y))
        x_change = np.max(np.abs(X_arr - prev_X))
        if y_change < tol and x_change < tol:
            break

    return y_arr, X_arr


def _within_transform(z: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract group means from z."""
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    unique = np.unique(groups)
    result = z.copy()
    for g in unique:
        mask = groups == g
        if np.sum(mask) > 0:
            result[mask] = z[mask] - z[mask].mean(axis=0)
    return result.ravel() if z.shape[1] == 1 else result


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