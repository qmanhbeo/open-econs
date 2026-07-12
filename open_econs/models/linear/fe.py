from typing import Any

import numpy as np
import pandas as pd

from open_econs.core.call_capture import capture_call as _capture_call
from open_econs._internal import errors
from open_econs.core.results import OLSResult
from open_econs.core.cov_type import validate_cov_type


def fe(
    formula: str,
    data: pd.DataFrame,
    entity: str | None = None,
    time: str | None = None,
    cluster: str | None = None,
    cov_type: str = "HC2",
    lags: int | None = None,
    hac_adjust: bool = False,
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
        Column name for cluster-robust standard errors. Takes precedence over
        ``cov_type="HAC"`` (cluster-robust is used when both are given).
    cov_type : str, default "HC2"
        Covariance estimator type. Used when *cluster* is not set. Common
        choices: ``"HC0"``–``"HC3"``, ``"nonrobust"``. Set ``cov_type="HAC"``
        to use Newey-West (1987) panel-HAC standard errors: the score
        contributions ``x_it * e_it`` are aggregated *within each time period*
        across entities, then a Bartlett-kernel long-run variance is applied
        *across* time periods (the Arellano / Driscoll-Kraay convention,
        matching statsmodels ``cov_nw_groupsum`` and Stata ``xtscc``). This
        requires *time* (which doubles as the time fixed-effects dimension —
        passing it incurs two-way FE) and *lags*.
    lags : int, optional
        Number of lags for Newey-West HAC (required when ``cov_type="HAC"``).
    hac_adjust : bool, default False
        Degrees-of-freedom correction for Newey-West HAC standard errors.

        When ``True``, the HAC variance is multiplied by ``N / (N - K)``
        (N = observations, K = number of regressors, intercept dropped). This
        is the N/(N-K) correction borrowed from White's HC1 and applied
        unconditionally by Stata's ``newey``. The original Newey & West (1987)
        paper does **not** include this correction.

        **Implementation comparison:**
        ================================ =================== ==============
        Implementation                    Applies N/(N-K)?    Default
        ================================ =================== ==============
        Newey & West (1987)               No                  —
        **Open-econs** (current)          **No**              **``False``**
        Statsmodels ``cov_nw_groupsum``   No                  ``use_correction=0``
        R ``sandwich::NeweyWest()``       No                  ``adjust=FALSE``
        Stata ``newey``                   Yes                 Always (no opt-out)
        MATLAB ``hac``                    Yes                 Default
        ================================ =================== ==============

        Set ``hac_adjust=True`` for SEs that match Stata. Leave ``False``
        (default) for the original NW1987 formula.

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
        lags=lags, hac_adjust=hac_adjust,
    )

    cov_type = validate_cov_type(
        cov_type,
        accepted={"nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"},
        estimator="fe()",
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

    # Identify intercept column (all zeros after demeaning) and non-intercept
    # columns.  We drop the intercept before fitting so that statsmodels gets
    # the correct rank and df, then apply the panel df correction ourselves.
    keep_mask = np.array([c != "Intercept" for c in XX.columns])
    kept_columns = [c for c in XX.columns if c != "Intercept"]

    if entity is not None and time is not None:
        entity_arr = data.loc[XX.index, entity].values
        time_arr = data.loc[XX.index, time].values
        y_arr, X_arr = _demean_two_way(y_arr, X_arr, entity_arr, time_arr)
    elif entity is not None:
        entity_arr = data.loc[XX.index, entity].values
        y_arr = _demean(y_arr, entity_arr)
        X_arr = _demean(X_arr, entity_arr)
    else:
        time_arr = data.loc[XX.index, time].values
        y_arr = _demean(y_arr, time_arr)
        X_arr = _demean(X_arr, time_arr)

    # Drop the (now all-zero) intercept column before fitting
    X_arr = X_arr[:, keep_mask]

    # Count absorbed FE groups for panel df correction.
    n_absorbed = 0
    if entity is not None:
        n_absorbed += len(np.unique(entity_arr))
    if time is not None:
        n_absorbed += len(np.unique(time_arr))
    if entity is not None and time is not None:
        n_absorbed -= 1  # avoid double-counting the grand mean

    import statsmodels.api as sm

    # V_cov / se_arr are computed explicitly for the HAC path; for cluster and
    # plain covariance types they are taken from the statsmodels fit below.
    V_cov = None
    se_arr = None

    if cluster is not None:
        if cluster not in data.columns:
            raise errors.cluster_column_error(cluster, data.columns.tolist())
        aligned_groups = data.loc[XX.index, cluster]
        X_df = pd.DataFrame(X_arr, columns=kept_columns)
        fitted = sm.OLS(y_arr, X_df).fit(
            cov_type="cluster",
            cov_kwds={"groups": aligned_groups},
        )
        cov_label = f"cluster({cluster})"
    else:
        use_hac = cov_type == "HAC"
        if use_hac:
            if lags is None:
                raise ValueError("Newey-West HAC requires `lags` (e.g. lags=1).")
            if time is None:
                raise ValueError(
                    "FE Newey-West HAC requires `time` (the time fixed-effects "
                    "dimension is used as the Newey-West period index)."
                )
            from open_econs.core.cov import newey_west_cov, _as_int_labels

            X_df = pd.DataFrame(X_arr, columns=kept_columns)
            fitted = sm.OLS(y_arr, X_df).fit(cov_type="nonrobust")
            time_labels = _as_int_labels(data.loc[XX.index, time].values)
            # Period-aggregation Newey-West (Arellano / Driscoll-Kraay): aggregate
            # score contributions within each time period, then Bartlett-kernel
            # HAC across periods.  cluster=time_labels is the within-panel HAC
            # convention (matches statsmodels cov_nw_groupsum).
            V_cov = newey_west_cov(
                X_arr, np.asarray(fitted.resid), max_lags=lags, cluster=time_labels,
                adjust=hac_adjust,
            )
            se_arr = np.sqrt(np.maximum(np.diag(V_cov), 0.0))
            cov_label = f"HAC({lags})"
        else:
            X_df = pd.DataFrame(X_arr, columns=kept_columns)
            fitted = sm.OLS(y_arr, X_df).fit(cov_type=cov_type)
            cov_label = cov_type

    n = int(fitted.nobs)
    k = X_arr.shape[1]  # number of regressors
    df_resid_adj = max(n - n_absorbed - k, 1)
    df_model_adj = int(fitted.df_model)

    coef_arr = np.asarray(fitted.params)
    if se_arr is None:
        se_arr = np.asarray(fitted.bse)
    if V_cov is None:
        V_cov = np.asarray(fitted.cov_params())
    t_arr = np.asarray(fitted.tvalues)
    p_arr = np.asarray(fitted.pvalues)
    conf_arr = np.asarray(fitted.conf_int())

    # Rescale SEs, t-stats, p-values for the corrected df.  For non-robust
    # (and HC1) covariances the SE is proportional to sqrt(SSR / df), so
    # scaling by sqrt(df_old / df_new) is exact.  For cluster-robust SEs
    # the same approximation is standard practice (Stata's xtreg, fe does
    # the same).
    df_old = max(int(fitted.df_resid), 1)
    _cov = None
    if df_resid_adj != df_old and df_old > 0:
        scale = np.sqrt(df_old / df_resid_adj)
        se_arr = se_arr * scale
        from scipy import stats as _stats
        t_arr = coef_arr / se_arr
        p_arr = 2.0 * _stats.t.sf(np.abs(t_arr), df_resid_adj)
        crit = _stats.t.ppf(0.975, df_resid_adj)
        conf_arr = np.column_stack([coef_arr - crit * se_arr, coef_arr + crit * se_arr])
        # Store the df-scaled covariance so vcov() is consistent with
        # the reported standard errors.  The raw covariance (HAC or
        # statsmodels) uses sigma2 = SSR/(N-k); Stata uses SSR/(N-g-k).
        # The ratio is df_old/df_resid_adj, applied element-wise.
        _cov = pd.DataFrame(
            V_cov * (df_old / df_resid_adj),
            index=kept_columns,
            columns=kept_columns,
        )

    conf_int = pd.DataFrame(
        {"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]},
        index=kept_columns,
    )

    ssr = float(np.sum(fitted.resid ** 2))

    # Within R-squared: denominator is the SST of the within-transformed y
    # using only the *entity* (or time) demeaning — matching Stata's e(r2_w).
    # For one-way FE this is just sum(y_dm^2).  For two-way FE we use the
    # entity-only demeaned y so that R² measures the share of within-entity
    # variation explained (Stata's convention).
    if entity is not None:
        y_for_r2 = _demean(yy.values.ravel().astype(float), entity_arr)
    elif time is not None:
        y_for_r2 = _demean(yy.values.ravel().astype(float), time_arr)
    else:
        y_for_r2 = yy.values.ravel().astype(float)
    sst = float(np.sum((y_for_r2 - np.mean(y_for_r2)) ** 2))
    r2 = 1.0 - ssr / (sst + 1e-15)
    if np.isnan(r2) or r2 < 0 or r2 > 1:
        r2 = float(fitted.rsquared)

    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(df_resid_adj, 1)

    fitted_values = pd.Series(fitted.fittedvalues, index=XX.index, name="fitted")
    residuals = pd.Series(fitted.resid, index=XX.index, name="residuals")

    rhs_formula = formula.split("~", 1)[1].strip()

    result = OLSResult(
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
        rsd=float(np.sqrt(ssr / max(df_resid_adj, 1))),
        llf=_safe_llf(fitted),
        aic=_safe_aic(fitted),
        bic=_safe_bic(fitted),
        fitted=fitted_values,
        residuals=residuals,
        call=call,
        condition_number=float(np.linalg.cond(X_arr)) if X_arr.shape[1] > 0 else 0.0,
        _X=XX,
        _fit=fitted,
    )
    # Attach the df-scaled covariance matrix so vcov() returns values
    # consistent with the panel-adjusted standard errors.
    if _cov is not None:
        object.__setattr__(result, "_cov", _cov)
    return result


def _demean(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """One-way within transform via O(n) group-mean subtraction.

    Subtracts each observation's group mean (the analytical solution to the
    dummy-regression projection) instead of forming the full dummy matrix and
    running a least-squares solve — which would cost O(n x G) memory for G
    groups and is the bottleneck for large panels.
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    cols = [f"c{i}" for i in range(y.shape[1])]
    s = pd.DataFrame(y, columns=cols)
    s["__g"] = groups
    means = s[cols].groupby(s["__g"]).transform("mean").values
    resid = y - means
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
    """Subtract group means from z (vectorized via pandas groupby)."""
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    cols = [f"c{i}" for i in range(z.shape[1])]
    df = pd.DataFrame(z, columns=cols)
    df["__g"] = groups
    means = df.groupby("__g")[cols].transform("mean").values
    return (z - means).ravel() if z.shape[1] == 1 else z - means


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