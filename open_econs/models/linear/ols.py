import re as _re
from typing import Any

import numpy as np
import pandas as pd

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
        leverage-adjusted small-sample; note: Stata's ``regress, robust``
        uses HC1), ``"HC1"`` (classic White SE), ``"HC0"``, ``"HC3"``,
        ``"nonrobust"``.  Ignored when *cluster* is provided (cluster-robust
        is used instead).

        Set ``cov_type="HAC"`` to use Newey-West (1987) heteroskedasticity-
        and autocorrelation-robust standard errors; the number of lags is
        given by *lags* and the time ordering by *time* (or *cluster* for
        panel HAC).

        **Why HC2 is the default (not ``nonrobust``):**

        OE defaults to HC2 rather than Stata's bare ``nonrobust`` default
        because defaulting to non-robust standard errors is widely considered
        poor applied practice — it silently understates uncertainty under
        heteroskedasticity, the single most common applied-econometrics
        footgun.  Users who want exact Stata-default parity should pass
        ``cov_type='nonrobust'`` explicitly.  This preserves both the parity
        promise (the *option* to exactly match Stata is one keyword away,
        fully documented) and sound econometric defaults (robust-by-default
        protects users who don't think to ask for it).

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

    Notes
    -----
    This function uses ``pyfixest.feols`` as the compute backend for
    non-HAC covariance types (nonrobust, HC0, HC1, HC2, HC3, CRV1).
    The HAC path retains the original ``statsmodels``-based implementation
    via ``open_econs.core.cov.newey_west_cov``.

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

    if model_spec is not None:
        stored_spec = model_spec.model_spec.rhs
    else:
        stored_spec = None

    _check_collinearity(XX)

    clusters = list(cluster) if isinstance(cluster, (list, tuple)) else None
    multiway = clusters is not None and len(clusters) > 1
    use_hac = cov_type == "HAC"

    if use_hac:
        return _ols_hac_path(
            formula=formula,
            formula_obj=formula_obj,
            model_spec=model_spec,
            XX=XX,
            yy=yy,
            data=data,
            cluster=cluster,
            lags=lags,
            time=time,
            hac_adjust=hac_adjust,
            original_n=original_n,
            dropped=dropped,
            cols_with_nas=cols_with_nas,
            call=call,
            rhs_formula=rhs_formula,
            stored_spec=stored_spec,
        )

    if multiway:
        assert clusters is not None
        return _ols_multiway_cluster_path(
            formula=formula,
            XX=XX,
            yy=yy,
            data=data,
            clusters=clusters,
            original_n=original_n,
            dropped=dropped,
            cols_with_nas=cols_with_nas,
            call=call,
            rhs_formula=rhs_formula,
            stored_spec=stored_spec,
        )

    return _ols_pyfixest_path(
        formula=formula,
        XX=XX,
        yy=yy,
        data=data,
        cluster=cluster if isinstance(cluster, str) else None,
        cov_type=cov_type,
        weights=weights,
        original_n=original_n,
        dropped=dropped,
        cols_with_nas=cols_with_nas,
        call=call,
        rhs_formula=rhs_formula,
        stored_spec=stored_spec,
    )


reg = ols


def _ols_pyfixest_path(
    *,
    formula: str,
    XX: pd.DataFrame,
    yy: pd.Series,
    data: pd.DataFrame,
    cluster: str | None,
    cov_type: str,
    weights: str | np.ndarray | pd.Series | None,
    original_n: int,
    dropped: int,
    cols_with_nas: list[str],
    call: dict[str, Any],
    rhs_formula: str,
    stored_spec: Any,
) -> OLSResult:
    """Run OLS through pyfixest and translate to OLSResult."""
    import pyfixest as pf

    y_name = yy.columns[0] if hasattr(yy, "columns") else "y"

    # Detect formulaic constructs that pyfixest handles natively (C(), i(), I()).
    # When these are present, pyfixest can process the original formula
    # directly — both formulaic and pyfixest agree on factor expansion and
    # stateful transforms.  For plain numeric formulas, we build the pyfixest
    # formula from XX.columns to avoid the two-parser divergence on `- 1`
    # (no intercept).
    _has_formulaic_factors = bool(
        _re.search(r'\b[CiI]\s*\(', formula.split("~", 1)[-1])
        if "~" in formula else False
    )

    has_intercept = "Intercept" in XX.columns

    if _has_formulaic_factors:
        # Pass the original formula through to pyfixest.  Build `work` from
        # the original data columns (pyfixest will re-expand C() itself).
        pf_fml = formula
        work = data.loc[XX.index].copy()
        if y_name not in work.columns:
            work[y_name] = yy.values.ravel()
    else:
        # Build pyfixest formula from XX.columns (post-formulaic model matrix).
        # We do NOT pass the original formula string through to pyfixest's
        # parser because two independent parsers (formulaic vs pyfixest's own)
        # disagree on `- 1` (no intercept) handling. Building from XX.columns
        # is the same proven pattern fe.py uses — it guarantees pyfixest sees
        # exactly the columns that formulaic already produced.
        x_part = " + ".join(c for c in XX.columns if c != "Intercept")
        pf_fml = f"{y_name} ~ {x_part}"
        if not has_intercept:
            pf_fml += " - 1"
        # Build work from XX columns (post-formulaic model matrix).
        work = pd.DataFrame(index=XX.index)
        for c in XX.columns:
            work[c] = XX[c].values
        work[y_name] = yy.values.ravel()

    # Merge in additional columns (cluster, weights) from data when needed.
    extra_cols = []
    if isinstance(cluster, str):
        extra_cols.append(cluster)
    if isinstance(weights, str):
        extra_cols.append(weights)
    for c in extra_cols:
        if c not in work.columns and c in data.columns:
            work[c] = data.loc[XX.index, c].values

    pf_vcov: Any
    if isinstance(cluster, str):
        if cluster not in data.columns:
            raise errors.cluster_column_error(cluster, data.columns.tolist())
        pf_vcov = {"CRV1": cluster}
        cov_label = f"cluster({cluster})"
    else:
        if cov_type == "nonrobust":
            pf_vcov = "iid"
        elif cov_type == "HC0":
            # pyfixest uses "hetero" for what's commonly called HC0
            pf_vcov = "hetero"
        else:
            pf_vcov = cov_type
        cov_label = cov_type

    pf_kwargs: dict[str, Any] = {"fml": pf_fml, "data": work, "vcov": pf_vcov}
    if weights is not None:
        if isinstance(weights, str):
            if weights not in data.columns:
                raise errors.missing_column_error(weights, data.columns.tolist())
            w_arr = work[weights].values.astype(float)
        else:
            w_arr = np.asarray(weights, dtype=float)
            if len(w_arr) != original_n:
                raise ValueError(
                    f"weights array length ({len(w_arr)}) does not match data rows ({original_n})"
                )
            w_arr = w_arr[XX.index]
        if np.any(w_arr < 0):
            raise ValueError("Weights must be non-negative.")
        work["_oe_weights"] = w_arr
        pf_kwargs["weights"] = "_oe_weights"

    # pyfixest's SSC computation divides by (N - df_k), which is undefined
    # for very small datasets (N ≤ k). Fall back to statsmodels for these
    # edge cases.
    n_after_drops = len(work)
    k_params = len(XX.columns)
    _use_pyfixest = (n_after_drops - k_params) > 0

    if _use_pyfixest:
        try:
            fit = pf.feols(**pf_kwargs)
        except (ZeroDivisionError, ValueError):
            _use_pyfixest = False

    if not _use_pyfixest:
        return _ols_statsmodels_fallback(
            formula=formula,
            XX=XX,
            yy=yy,
            work=work,
            pf_fml=pf_fml,
            cov_type=cov_type,
            cov_label=cov_label,
            original_n=original_n,
            dropped=dropped,
            cols_with_nas=cols_with_nas,
            call=call,
            rhs_formula=rhs_formula,
            stored_spec=stored_spec,
        )

    coef_dict = fit.coef().to_dict()
    se_dict = fit.se().to_dict()
    tstat_dict = fit.tstat().to_dict()
    pvalue_dict = fit.pvalue().to_dict()
    ci_df = fit.confint()

    all_columns = list(XX.columns)
    n_cols = len(all_columns)
    coef_arr = np.array([coef_dict.get(c, 0.0) for c in all_columns])
    se_arr = np.array([se_dict.get(c, np.nan) for c in all_columns])
    t_arr = np.array([tstat_dict.get(c, np.nan) for c in all_columns])
    p_arr = np.array([pvalue_dict.get(c, np.nan) for c in all_columns])

    conf_lower = np.array([ci_df.loc[c, ci_df.columns[0]] if c in ci_df.index else np.nan for c in all_columns])
    conf_upper = np.array([ci_df.loc[c, ci_df.columns[1]] if c in ci_df.index else np.nan for c in all_columns])
    conf_arr = np.column_stack([conf_lower, conf_upper])

    n = int(fit._N)
    k = n_cols
    df_resid = max(n - k, 1)
    df_model = k - 1 if has_intercept else k

    r2 = float(fit._r2)
    adj_r2 = float(fit._adj_r2)

    residuals = fit.resid()
    fitted_values = fit.predict()
    ssr = float(np.sum(residuals ** 2))
    sst = float(np.sum((yy.values.ravel() - np.mean(yy.values.ravel())) ** 2))
    r2_check = 1.0 - ssr / (sst + 1e-15)
    if np.isfinite(r2_check) and 0 <= r2_check <= 1:
        r2 = r2_check
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(df_resid, 1)

    f_stat, f_pval = _compute_f_stat(fit, n, k, has_intercept=has_intercept)

    llf = _compute_log_likelihood(residuals, n, k)
    aic_val = _compute_aic(llf, k)
    bic_val = _compute_bic(llf, k, n)

    condition_number = float(np.linalg.cond(XX.values)) if XX.shape[1] > 0 else 0.0

    _cov = pd.DataFrame(
        fit._vcov,
        index=all_columns,
        columns=all_columns,
    )

    result = OLSResult(
        formula=formula,
        rhs_formula=rhs_formula,
        nobs=n,
        df_resid=df_resid,
        df_model=df_model,
        cov_type=cov_label,
        coefficients=pd.Series(coef_arr, index=all_columns),
        std_errors=pd.Series(se_arr, index=all_columns),
        t_stats=pd.Series(t_arr, index=all_columns),
        p_values=pd.Series(p_arr, index=all_columns),
        conf_int=pd.DataFrame({"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]}, index=all_columns),
        r_squared=r2,
        adj_r_squared=adj_r2,
        f_statistic=f_stat,
        f_p_value=f_pval,
        rsd=float(np.sqrt(ssr / max(df_resid, 1))),
        llf=llf,
        aic=aic_val,
        bic=bic_val,
        fitted=pd.Series(fitted_values, index=XX.index, name="fitted"),
        residuals=pd.Series(residuals, index=XX.index, name="residuals"),
        call=call,
        model_spec=stored_spec,
        condition_number=condition_number,
        _X=XX,
        _fit=None,
    )
    object.__setattr__(result, "_cov", _cov)
    return result


def _ols_statsmodels_fallback(
    *,
    formula: str,
    XX: pd.DataFrame,
    yy: pd.Series,
    work: pd.DataFrame,
    pf_fml: str,
    cov_type: str,
    cov_label: str,
    original_n: int,
    dropped: int,
    cols_with_nas: list[str],
    call: dict[str, Any],
    rhs_formula: str,
    stored_spec: Any,
) -> OLSResult:
    """Fallback to statsmodels for edge cases pyfixest cannot handle."""
    import statsmodels.api as sm

    y_arr = yy.values.ravel()
    fitted = sm.OLS(y_arr, XX.values).fit(cov_type="nonrobust")
    coef_arr = fitted.params

    # Recompute with the requested vcov
    if cov_type == "nonrobust":
        V = fitted.cov_params()
    else:
        sm_cov_map = {"HC0": "HC0", "HC1": "HC1", "HC2": "HC2", "HC3": "HC3"}
        sm_type = sm_cov_map.get(cov_type, "HC1")
        fitted_robust = sm.OLS(y_arr, XX.values).fit(cov_type=sm_type)
        V = fitted_robust.cov_params()
        coef_arr = fitted_robust.params

    se_arr = np.sqrt(np.maximum(np.diag(V), 0.0))
    t_arr = np.where(se_arr > 0, coef_arr / se_arr, np.nan)
    from scipy.stats import norm as _norm
    p_arr = 2.0 * (1.0 - _norm.cdf(np.abs(t_arr)))
    conf_arr = np.column_stack(
        [coef_arr - 1.96 * se_arr, coef_arr + 1.96 * se_arr]
    )

    conf_int = pd.DataFrame(
        {"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]},
        index=XX.columns,
    )
    fitted_values = pd.Series(fitted.fittedvalues, index=XX.index, name="fitted")
    residuals = pd.Series(fitted.resid, index=XX.index, name="residuals")

    n = int(fitted.nobs)
    k = XX.shape[1]
    has_intercept = "Intercept" in XX.columns
    df_resid = int(fitted.df_resid)
    df_model = k - 1 if has_intercept else k

    f_stat = _safe_fvalue(fitted)
    f_pval = _safe_f_pvalue(fitted)

    condition_number = float(np.linalg.cond(XX.values)) if XX.shape[1] > 0 else 0.0

    result = OLSResult(
        formula=formula,
        rhs_formula=rhs_formula,
        nobs=n,
        df_resid=df_resid,
        df_model=df_model,
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
    object.__setattr__(
        result,
        "_cov",
        pd.DataFrame(V, index=XX.columns, columns=XX.columns),
    )
    return result


def _ols_hac_path(
    *,
    formula: str,
    formula_obj: Any,
    model_spec: Any,
    XX: pd.DataFrame,
    yy: pd.Series,
    data: pd.DataFrame,
    cluster: str | list[str] | None,
    lags: int | None,
    time: str | None,
    hac_adjust: bool,
    original_n: int,
    dropped: int,
    cols_with_nas: list[str],
    call: dict[str, Any],
    rhs_formula: str,
    stored_spec: Any,
) -> OLSResult:
    """Run OLS HAC via statsmodels/newey_west_cov (retained from original)."""
    import statsmodels.api as sm

    y_arr = yy.values.ravel()

    if lags is None:
        raise ValueError("Newey-West HAC requires `lags` (e.g. lags=1).")

    fitted = sm.OLS(y_arr, XX.values).fit(cov_type="nonrobust")
    coef_arr = fitted.params

    from open_econs.core.cov import newey_west_cov, _as_int_labels

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

    conf_int = pd.DataFrame(
        {"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]},
        index=XX.columns,
    )
    fitted_values = pd.Series(fitted.fittedvalues, index=XX.index, name="fitted")
    residuals = pd.Series(fitted.resid, index=XX.index, name="residuals")

    n = int(fitted.nobs)
    k = XX.shape[1]
    df_resid = max(n - k, 1)

    f_stat = _safe_fvalue(fitted)
    f_pval = _safe_f_pvalue(fitted)

    condition_number = float(np.linalg.cond(XX.values)) if XX.shape[1] > 0 else 0.0

    result = OLSResult(
        formula=formula,
        rhs_formula=rhs_formula,
        nobs=n,
        df_resid=df_resid,
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
    object.__setattr__(
        result,
        "_cov",
        pd.DataFrame(V, index=XX.columns, columns=XX.columns),
    )
    return result


def _ols_multiway_cluster_path(
    *,
    formula: str,
    XX: pd.DataFrame,
    yy: pd.Series,
    data: pd.DataFrame,
    clusters: list[str],
    original_n: int,
    dropped: int,
    cols_with_nas: list[str],
    call: dict[str, Any],
    rhs_formula: str,
    stored_spec: Any,
) -> OLSResult:
    """Run OLS with multi-way clustering via Cameron-Gelbach-Miller (2011)."""
    import statsmodels.api as sm
    from open_econs.core.cov import multiway_cluster_cov, _as_int_labels

    y_arr = yy.values.ravel()

    fitted = sm.OLS(y_arr, XX.values).fit(cov_type="nonrobust")
    coef_arr = fitted.params

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

    conf_int = pd.DataFrame(
        {"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]},
        index=XX.columns,
    )
    fitted_values = pd.Series(fitted.fittedvalues, index=XX.index, name="fitted")
    residuals = pd.Series(fitted.resid, index=XX.index, name="residuals")

    n = int(fitted.nobs)
    k = XX.shape[1]
    df_resid = max(n - k, 1)

    f_stat = _safe_fvalue(fitted)
    f_pval = _safe_f_pvalue(fitted)

    condition_number = float(np.linalg.cond(XX.values)) if XX.shape[1] > 0 else 0.0

    result = OLSResult(
        formula=formula,
        rhs_formula=rhs_formula,
        nobs=n,
        df_resid=df_resid,
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
    object.__setattr__(
        result,
        "_cov",
        pd.DataFrame(V, index=XX.columns, columns=XX.columns),
    )
    return result


def _compute_f_stat(
    fit: Any,
    n: int,
    k: int,
    has_intercept: bool,
) -> tuple[float, float]:
    """Compute Wald F-statistic and its p-value using the model's VCE.

    This matches Stata's ``regress`` behaviour:
    - Under ``vce(ols)``: uses the iid VCE → equivalent to the ANOVA F.
    - Under ``vce(robust)`` / ``vce(cluster)`` / HC*: uses the robustly
      estimated VCE (a Wald test on the slope coefficients).

    The F-statistic is defined as::

        F = (R b_hat)' (R V R')^{-1} (R b_hat) / q

    where R selects the slope coefficients (excluding the intercept) and
    q = number of slope restrictions.  When there is no intercept, R = I_k
    and q = k (testing all coefficients = 0).

    References
    ----------
    - Stata Manuals: ``regress`` (Methods and Formulas section confirms the
      Wald-form F under ``vce(robust)``).
    - Davidson & MacKinnon (1993), *Estimation and Inference in Econometrics*,
      Ch. 7.
    """
    try:
        from scipy.stats import f as _f_dist

        b = fit.coef().values
        V = fit._vcov
        k_total = len(b)

        if k_total <= 1:
            return float("nan"), float("nan")

        # R selects slope coefficients: skip index 0 (intercept) if present.
        if has_intercept:
            q = k_total - 1
            R = np.zeros((q, k_total))
            for i in range(q):
                R[i, i + 1] = 1.0
        else:
            q = k_total
            R = np.eye(k_total)

        Rb = R @ b
        RVR = R @ V @ R.T
        W = float(Rb @ np.linalg.inv(RVR) @ Rb)
        f_stat = W / q

        dfd = max(n - k_total, 1)
        f_pval = float(_f_dist.sf(f_stat, q, dfd))
        return float(f_stat), f_pval
    except Exception:
        return float("nan"), float("nan")


def _compute_log_likelihood(residuals: np.ndarray, n: int, k: int) -> float:
    """Compute log-likelihood for OLS from residuals."""
    try:
        ssr = float(np.sum(residuals ** 2))
        sigma2 = ssr / n
        if sigma2 <= 0:
            return float("nan")
        llf = -n / 2.0 * (np.log(2 * np.pi) + np.log(sigma2) + 1)
        return float(llf)
    except Exception:
        return float("nan")


def _compute_aic(llf: float, k: int) -> float:
    """Compute AIC from log-likelihood."""
    if not np.isfinite(llf):
        return float("nan")
    return float(2 * k - 2 * llf)


def _compute_bic(llf: float, k: int, n: int) -> float:
    """Compute BIC from log-likelihood."""
    if not np.isfinite(llf):
        return float("nan")
    return float(k * np.log(n) - 2 * llf)


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
