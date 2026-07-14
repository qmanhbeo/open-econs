from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from open_econs.core.call_capture import capture_call as _capture_call
from open_econs._internal import errors
from open_econs._internal.errors import VcovTypeNotSupportedError
from open_econs.core.results import OLSResult
from open_econs.core.cov_type import validate_cov_type


def fe(
    formula: str,
    data: pd.DataFrame,
    entity: str | None = None,
    time: str | None = None,
    fixed_effects: list[str] | None = None,
    cluster: str | list[str] | None = None,
    cov_type: str = "HC1",
    lags: int | None = None,
    hac_adjust: bool = False,
) -> OLSResult:
    """Estimate a linear fixed-effects (within) model.

    Parameters
    ----------
    formula : str
        Two-sided formula string, e.g. ``"y ~ x1 + x2"``. Do *not* include
        the fixed-effect indicator in the formula; use *entity*/*time* or
        *fixed_effects*.
    data : pd.DataFrame
        Data containing all variables referenced in *formula* plus the
        entity/time/fixed-effects columns.
    entity : str, optional
        Column name for entity (panel unit) fixed effects. If provided, both
        *y* and *X* are group-demeaned within each entity.
    time : str, optional
        Column name for time fixed effects. If *entity* is also provided,
        two-way fixed effects are used (entity and time dummies absorbed
        via iterative demeaning for unbalanced panels).
    fixed_effects : list of str, optional
        Column names for arbitrary N-way fixed effects. Takes precedence
        over *entity*/*time* when provided — pass **either** ``entity=``
        /``time=`` **or** ``fixed_effects=``, not both (raises ``ValueError``).

        Use this for 3+-way FE that cannot be expressed with just two named
        kwargs::

            # 3-way FE
            oe.fe("y ~ x", data=df, fixed_effects=["firm", "year", "industry"])
            # same as 2-way via entity/time
            oe.fe("y ~ x", data=df, entity="firm", time="year")

        Internally this maps to ``pyfixest``'s ``| f1 + f2 + ...`` syntax.
    cluster : str or list of str, optional
        Column name(s) for cluster-robust standard errors. Takes precedence
        over ``cov_type="HAC"`` (cluster-robust is used when both are given).
        Passing a *list* requests multi-way clustering (e.g.
        ``["firm", "year"]``).
    cov_type : str, default "HC1"
        Covariance estimator type. Used when *cluster* is not set. Common
        choices: ``"HC1"`` (default, matching Stata ``xtreg, fe``),
        ``"HC0"``, ``"nonrobust"``. Set ``cov_type="HAC"`` to use Newey-West
        (1987) panel-HAC standard errors.

        .. note::
           HC2 and HC3 are **not supported** for models with absorbed fixed
           effects (the leverage adjustments are invalid once FEs are absorbed).
           Use ``cov_type="HC1"`` instead.  This matches ``pyfixest``'s
           behaviour and is statistically correct.
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

    Notes
    -----
    This function uses ``pyfixest.feols`` as the compute backend for
    non-HAC covariance types (HC0, HC1, nonrobust, CRV1, CRV3).  The
    ``entity=``/``time=`` and ``fixed_effects=`` kwargs are mapped internally
    to ``pyfixest``'s ``| f1 + f2 + ...`` formula syntax.  The HAC path
    retains the original ``statsmodels``-based implementation.

    Arbitrary N-way FE (3+) are only supported on the pyfixest path (non-HAC).
    Passing ``cov_type="HAC"`` with more than two fixed-effect columns raises
    ``ValueError``.

    **Breaking changes** (v1.1):
    - The default ``cov_type`` changed from ``"HC2"`` to ``"HC1"``.  HC2/HC3
      are now forbidden for FE models because the leverage adjustments in
      HC2/HC3 are invalid once fixed effects are absorbed.  This matches
      ``pyfixest``'s convention.
    - Standard errors, t-statistics, p-values, and confidence intervals may
      differ slightly from previous versions due to the adoption of
      ``pyfixest``'s small-sample corrections (fixest-standard
      leverage-adjusted dof scaling).

    Examples
    --------
    >>> import open_econs as oe
    >>> r = oe.fe("y ~ x1 + x2", data=df, entity="country", time="year")
    >>> r.tidy()
    >>> # 3-way fixed effects
    >>> r = oe.fe("y ~ x1", data=df, fixed_effects=["firm", "year", "industry"])
    """
    call = _capture_call(
        formula=formula, entity=entity, time=time, fixed_effects=fixed_effects,
        cluster=cluster, cov_type=cov_type,
        lags=lags, hac_adjust=hac_adjust,
    )

    cov_type = validate_cov_type(
        cov_type,
        accepted={"nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"},
        estimator="fe()",
    )

    # D2: forbid HC2/HC3 on FE models (leverage adjustments are invalid once
    # FEs are absorbed; pyfixest also refuses these).
    if cov_type in ("HC2", "HC3"):
        raise VcovTypeNotSupportedError(cov_type)

    # ---- validate FE specification ----
    if fixed_effects is not None and (entity is not None or time is not None):
        raise ValueError(
            "Pass either fixed_effects= OR entity=/time=, not both. "
            "fixed_effects= takes precedence and ignores entity=/time=."
        )

    if fixed_effects is None and entity is None and time is None:
        raise ValueError(
            "At least one of fixed_effects= or entity= or time= must be provided."
        )

    # ---- build formula and data for pyfixest ----
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

    # ---- determine the FE columns ----
    if fixed_effects is not None:
        fe_parts = list(fixed_effects)
    else:
        fe_parts: list[str] = []
        if entity is not None:
            fe_parts.append(entity)
        if time is not None:
            fe_parts.append(time)

    fe_formula_suffix = " + ".join(fe_parts)

    # ---- HAC path: retain the original statsmodels implementation ----
    # (pyfixest does not support HAC with absorbed FE)
    use_hac = cov_type == "HAC"
    use_cluster = cluster is not None

    if use_hac and len(fe_parts) > 2:
        raise ValueError(
            f"HAC standard errors are not supported with {len(fe_parts)}-way "
            f"fixed effects ({', '.join(fe_parts)}). The HAC path only supports "
            f"up to 2-way FE. Use cov_type='HC1' or cluster=... instead."
        )

    if use_hac:
        # Map fixed_effects to entity/time for the HAC path (max 2-way guarded above).
        hac_entity = entity
        hac_time = time
        if fixed_effects is not None:
            hac_entity = fixed_effects[0] if len(fixed_effects) >= 1 else None
            hac_time = fixed_effects[1] if len(fixed_effects) >= 2 else None

        return _fe_hac_path(
            formula=formula,
            formula_obj=formula_obj,
            model_spec=model_spec,
            XX=XX,
            yy=yy,
            data=data,
            entity=hac_entity,
            time=hac_time,
            fe_parts=fe_parts,
            lags=lags,
            hac_adjust=hac_adjust,
            original_n=original_n,
            dropped=dropped,
            call=call,
        )

    # ---- pyfixest path: nonrobust / HC0 / HC1 / cluster ----
    return _fe_pyfixest_path(
        formula=formula,
        fe_formula_suffix=fe_formula_suffix,
        data=data,
        XX=XX,
        yy=yy,
        entity=entity,
        time=time,
        fixed_effects=fixed_effects,
        cluster=cluster,
        cov_type=cov_type,
        original_n=original_n,
        dropped=dropped,
        call=call,
    )


def _fe_pyfixest_path(
    *,
    formula: str,
    fe_formula_suffix: str,
    data: pd.DataFrame,
    XX: pd.DataFrame,
    yy: pd.Series,
    entity: str | None,
    time: str | None,
    fixed_effects: list[str] | None,
    cluster: str | list[str] | None,
    cov_type: str,
    original_n: int,
    dropped: int,
    call: dict[str, Any],
) -> OLSResult:
    """Run the FE estimation through pyfixest and translate to OLSResult."""
    import pyfixest as pf

    y_name = yy.columns[0] if hasattr(yy, "columns") else "y"
    # Build the working dataframe with the LHS and RHS columns plus FE/cluster.
    needed_cols = list(XX.columns)
    if fixed_effects is not None:
        needed_cols.extend(fixed_effects)
    else:
        if entity is not None:
            needed_cols.append(entity)
        if time is not None:
            needed_cols.append(time)
    if isinstance(cluster, str):
        needed_cols.append(cluster)
    elif isinstance(cluster, list):
        needed_cols.extend(cluster)
    # Deduplicate while preserving order (pyfixest / narwhals require unique columns).
    seen: set[str] = set()
    unique_cols: list[str] = []
    for c in needed_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)
    work = data.loc[XX.index, [c for c in unique_cols if c in data.columns]].copy()
    work[y_name] = yy.values.ravel()

    # Build the pyfixest formula:  "y ~ x1 + x2 | entity + time"
    x_part = " + ".join(c for c in XX.columns if c != "Intercept")
    pf_fml = f"{y_name} ~ {x_part} | {fe_formula_suffix}"

    # Map cov_type to pyfixest vcov argument.
    pf_vcov: Any
    if cluster is not None:
        if isinstance(cluster, str):
            pf_vcov = {"CRV1": cluster}
        elif isinstance(cluster, list):
            pf_vcov = {"CRV1": " + ".join(cluster)}
        cov_label = f"cluster({cluster})" if isinstance(cluster, str) else "cluster(" + ", ".join(cluster) + ")"
    else:
        if cov_type == "nonrobust":
            pf_vcov = "iid"
        else:
            pf_vcov = cov_type  # HC0, HC1
        cov_label = cov_type

    fit = pf.feols(pf_fml, data=work, vcov=pf_vcov)

    # ---- extract results from pyfixest ----
    # pyfixest drops the intercept from coef(), se(), confint(), _vcov.
    # Match the old statsmodels path: kept_columns excludes "Intercept".
    all_columns = list(XX.columns)
    kept_columns = [c for c in all_columns if c != "Intercept"]

    coef_dict = fit.coef().to_dict()
    se_dict = fit.se().to_dict()
    tstat_dict = fit.tstat().to_dict()
    pvalue_dict = fit.pvalue().to_dict()
    ci_df = fit.confint()

    # pyfixest reports slopes only — map directly.
    n_cols = len(kept_columns)
    coef_arr = np.array([coef_dict.get(c, 0.0) for c in kept_columns])
    se_arr = np.array([se_dict.get(c, np.nan) for c in kept_columns])
    t_arr = np.array([tstat_dict.get(c, np.nan) for c in kept_columns])
    p_arr = np.array([pvalue_dict.get(c, np.nan) for c in kept_columns])

    conf_lower = np.array([ci_df.loc[c, ci_df.columns[0]] if c in ci_df.index else np.nan for c in kept_columns])
    conf_upper = np.array([ci_df.loc[c, ci_df.columns[1]] if c in ci_df.index else np.nan for c in kept_columns])
    conf_arr = np.column_stack([conf_lower, conf_upper])

    n = int(fit._N)
    k = len(kept_columns)  # slopes only (no intercept)

    # Count absorbed degrees of freedom. For 1-way FE it's simply the number
    # of groups; for 2+ way FE we need inclusion-exclusion on the union of
    # FE groups.
    if fixed_effects is not None:
        fe_cols_for_nabs = fixed_effects
    else:
        fe_cols_for_nabs = [c for c in [entity, time] if c is not None]

    n_absorbed = _count_absorbed_dof(data.loc[XX.index], fe_cols_for_nabs)

    df_resid_adj = max(n - n_absorbed - k, 1)
    df_model_adj = k

    r2 = float(fit._r2)
    adj_r2 = float(fit._adj_r2)

    # Within R-squared: use the first FE column for demeaning (entity convention).
    first_fe_col = fe_cols_for_nabs[0] if fe_cols_for_nabs else None
    if first_fe_col is not None:
        fe_arr_for_r2 = data.loc[XX.index, first_fe_col].values
        y_for_r2 = _demean(yy.values.ravel().astype(float), fe_arr_for_r2)
    else:
        y_for_r2 = yy.values.ravel().astype(float)
    ssr = float(np.sum(fit.resid() ** 2))
    sst = float(np.sum((y_for_r2 - np.mean(y_for_r2)) ** 2))
    r2_within = 1.0 - ssr / (sst + 1e-15)
    if np.isnan(r2_within) or r2_within < 0 or r2_within > 1:
        r2_within = r2

    # F-statistic from pyfixest.
    f_stat = float(fit._f_statistic) if fit._f_statistic is not None else float("nan")

    fitted_values = pd.Series(
        fit.predict(), index=XX.index, name="fitted",
    )
    residuals = pd.Series(fit.resid(), index=XX.index, name="residuals")

    rhs_formula = formula.split("~", 1)[1].strip()

    # Build covariance DataFrame for vcov().
    # pyfixest._vcov is (n_slope x n_slope), slopes only — same as kept_columns.
    _cov = pd.DataFrame(
        fit._vcov,
        index=kept_columns,
        columns=kept_columns,
    )

    condition_number = float(np.linalg.cond(XX.values)) if XX.shape[1] > 0 else 0.0

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
        conf_int=pd.DataFrame({"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]}, index=kept_columns),
        r_squared=float(r2_within),
        adj_r_squared=float(adj_r2),
        f_statistic=f_stat,
        f_p_value=float("nan"),  # TODO(pyfixest-gap): pyfixest does not expose F p-value for FE models
        rsd=float(np.sqrt(ssr / max(df_resid_adj, 1))),
        llf=float("nan"),  # TODO(pyfixest-gap): pyfixest does not expose log-likelihood for FE models
        aic=float("nan"),  # TODO(pyfixest-gap): pyfixest does not expose AIC for FE models
        bic=float("nan"),  # TODO(pyfixest-gap): pyfixest does not expose BIC for FE models
        fitted=fitted_values,
        residuals=residuals,
        call=call,
        condition_number=condition_number,
        _X=XX,
        _fit=None,
    )
    object.__setattr__(result, "_cov", _cov)
    return result


def _fe_hac_path(
    *,
    formula: str,
    formula_obj: Any,
    model_spec: Any,
    XX: pd.DataFrame,
    yy: pd.Series,
    data: pd.DataFrame,
    entity: str | None,
    time: str | None,
    fe_parts: list[str],
    lags: int | None,
    hac_adjust: bool,
    original_n: int,
    dropped: int,
    call: dict[str, Any],
) -> OLSResult:
    """Run the FE HAC estimation using the original statsmodels path."""
    import statsmodels.api as sm

    keep_mask = np.array([c != "Intercept" for c in XX.columns])
    kept_columns = [c for c in XX.columns if c != "Intercept"]
    y_arr = yy.values.ravel().astype(float)
    X_arr = XX.values.astype(float)

    entity_arr = None
    time_arr = None
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

    X_arr = X_arr[:, keep_mask]

    n_absorbed = _count_absorbed_dof(data.loc[XX.index], fe_parts)

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
    V_cov = newey_west_cov(
        X_arr, np.asarray(fitted.resid), max_lags=lags, cluster=time_labels,
        adjust=hac_adjust,
    )
    se_arr = np.sqrt(np.maximum(np.diag(V_cov), 0.0))
    cov_label = f"HAC({lags})"

    n = int(fitted.nobs)
    k = X_arr.shape[1]
    df_resid_adj = max(n - n_absorbed - k, 1)
    df_model_adj = int(fitted.df_model)

    coef_arr = np.asarray(fitted.params)
    t_arr = np.where(se_arr > 0, coef_arr / se_arr, np.nan)
    from scipy import stats as _stats
    p_arr = 2.0 * _stats.t.sf(np.abs(t_arr), df_resid_adj)
    crit = _stats.t.ppf(0.975, df_resid_adj)
    conf_arr = np.column_stack([coef_arr - crit * se_arr, coef_arr + crit * se_arr])

    _cov = None
    df_old = max(int(fitted.df_resid), 1)
    if df_resid_adj != df_old and df_old > 0:
        _cov = pd.DataFrame(
            V_cov * (df_old / df_resid_adj),
            index=kept_columns,
            columns=kept_columns,
        )
    else:
        _cov = pd.DataFrame(V_cov, index=kept_columns, columns=kept_columns)

    ssr = float(np.sum(fitted.resid ** 2))
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
        conf_int=pd.DataFrame({"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]}, index=kept_columns),
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
    if _cov is not None:
        object.__setattr__(result, "_cov", _cov)
    return result


def _demean(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """One-way within transform via O(n) group-mean subtraction."""
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
    """Iterative (alternating-projections) demeaning for two-way FE."""
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


def _count_absorbed_dof(df: pd.DataFrame, fe_cols: list[str]) -> int:
    """Count absorbed degrees of freedom for N-way FE.

    The absorbed DOF equals the dimension of the linear space spanned by all
    the FE indicator variables (including the intercept).  Since the intercept
    is in every FE subspace, it is counted ``len(fe_cols)`` times and we
    subtract ``len(fe_cols) - 1`` to correct.

    Formula: ``sum(n_groups_i) - (k - 1)`` where ``k = len(fe_cols)``.
    """
    if not fe_cols:
        return 0
    n_groups = sum(df[c].nunique() for c in fe_cols)
    return n_groups - (len(fe_cols) - 1)
