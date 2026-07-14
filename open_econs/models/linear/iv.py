from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from open_econs._version import __version__
from open_econs.core.call_capture import capture_call as _capture_call
from open_econs._internal import errors
from open_econs._internal.errors import VcovTypeNotSupportedError
from open_econs.core.base import BaseModel
from open_econs.core.cov_type import validate_cov_type


class IVResult(BaseModel):
    def __init__(
        self,
        *,
        formula: str,
        nobs: int,
        df_resid: int,
        df_model: int,
        cov_type: str,
        coefficients: pd.Series,
        std_errors: pd.Series,
        z_stats: pd.Series,
        p_values: pd.Series,
        conf_int: pd.DataFrame,
        rsd: float,
        first_stage_f: pd.Series,
        cragg_donald_f: float,
        hansen_j_stat: float,
        hansen_j_p: float,
        fitted: pd.Series,
        residuals: pd.Series,
        call: dict[str, Any],
        _fit: Any = None,
        _exog_names: list[str] | None = None,
        _endog_names: list[str] | None = None,
    ) -> None:
        self.formula = formula
        self.data_shape = (nobs, coefficients.shape[0])
        self.cov_type = cov_type
        self.call = call
        self.timestamp = datetime.now()
        self.package_version = __version__

        self.nobs = nobs
        self.df_resid = df_resid
        self.df_model = df_model
        self.coefficients = coefficients
        self.std_errors = std_errors
        self.z_stats = z_stats
        self.p_values = p_values
        self.conf_int = conf_int
        self.rsd = rsd
        self.first_stage_f = first_stage_f
        self.cragg_donald_stat = cragg_donald_f
        self.hansen_j_stat = hansen_j_stat
        self.hansen_j_p_value = hansen_j_p
        self.fitted_values = fitted if fitted is not None else pd.Series(dtype=float)
        self.residuals = residuals
        self._fit = _fit
        self._exog_var_names = _exog_names or []
        self._endog_var_names = _endog_names or []

        self._freeze()

    def tidy(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "Variable": self.coefficients.index,
            "Coef": self.coefficients.values,
            "Std Err": self.std_errors.values,
            "z": self.z_stats.values,
            "P>|z|": self.p_values.values,
            "0.025": self.conf_int["lower"].values,
            "0.975": self.conf_int["upper"].values,
        })
        df.index.name = None
        return df

    def summary(self) -> str:
        endog_name = self.formula.split("~")[0].strip()
        parts = self.formula.split("|", 1)
        instr_raw = parts[1].strip()
        fs_rows = []
        for var, fval in self.first_stage_f.items():
            fs_rows.append(f"  {var:<20s}  {fval:>8.4f}")

        fs_block = "\n".join(fs_rows)
        exog_str = ", ".join(self._exog_var_names) if self._exog_var_names else "none"
        endog_str = ", ".join(self._endog_var_names) if self._endog_var_names else "N/A"
        header = (
            f"                       IV-2SLS Regression Results                         \n"
            f"======================================================================\n"
            f"Dep. Variable:               {endog_name}\n"
            f"Exogenous:                   {exog_str}\n"
            f"Endogenous:                  {endog_str}\n"
            f"Instruments:                 {instr_raw}\n"
            f"No. Observations:            {self.nobs}\n"
            f"Df Residuals:                {self.df_resid}\n"
            f"Df Model:                    {self.df_model}\n"
            f"Covariance Type:             {self.cov_type}\n"
            f"======================================================================\n"
            f"First-stage F-statistics:\n{fs_block}\n"
            f"======================================================================\n"
            f"Cragg-Donald Wald F-stat:   {self.cragg_donald_stat:.4f}\n"
            f"Hansen J (overid) chi2:      {self.hansen_j_stat:.4f}\n"
            f"Hansen J p-value:            {self.hansen_j_p_value:.6e}\n"
            f"======================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return (
            header + tbl +
            "\n======================================================================\n"
        )

    def vcov(self) -> pd.DataFrame:
        """Return the 2SLS/IV parameter variance-covariance matrix as a DataFrame."""
        cov_df = getattr(self, "_cov", None)
        if cov_df is not None:
            common = [c for c in self.coefficients.index if c in cov_df.index]
            return cov_df.loc[common, common]
        if self._fit is not None:
            cov = np.asarray(self._fit.cov, dtype=float)
            return pd.DataFrame(
                cov,
                index=self.coefficients.index,
                columns=self.coefficients.index,
            )
        raise RuntimeError(
            "vcov() requires a fitted model result."
        )

    def first_stage(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Variable": self.first_stage_f.index,
            "F": self.first_stage_f.values,
        })


def iv(
    formula: str,
    data: pd.DataFrame,
    cov_type: str = "robust",
    cluster: str | list[str] | None = None,
    lags: int | None = None,
    time: str | None = None,
    hac_adjust: bool = False,
    entity: str | None = None,
    time_fe: str | None = None,
    fixed_effects: list[str] | None = None,
) -> IVResult:
    """Estimate an IV-2SLS regression.

    Parameters
    ----------
    formula : str
        Three-part formula ``y ~ x1 + w1 + w2 | x1 ~ z1 + z2``
        where variables left of ``~`` inside the RHS block are endogenous,
        and variables left of ``|`` but outside the ``~`` block are exogenous.

        Alternatively, the simpler ``y ~ x1 + w1 + w2 | z1 + z2`` form
        is accepted where **all** non-LHS variables are treated as endogenous
        (legacy, not recommended for applied use).

        Best practice: ``y ~ w1 + w2 | x1 ~ z1 + z2`` where ``w1, w2`` are
        exogenous controls, ``x1`` is the endogenous regressor, and
        ``z1, z2`` are instruments.
    data : pd.DataFrame
        Data containing all variables referenced in *formula*.
    cov_type : str, default "robust"
        Covariance estimator type.

        ``"nonrobust"`` -- unadjusted (iid) standard errors.
        ``"robust"`` / ``"HC1"`` -- HC1 robust standard errors (default,
        matching Stata ``ivregress 2sls, robust``).
        ``"HC0"`` -- HC0 (White) robust standard errors (without small-sample
        correction).  Only available without absorbed fixed effects.
        ``"HC2"`` / ``"HC3"`` -- higher-order HC corrections.  **Not supported
        for models with absorbed fixed effects** (D2: the leverage adjustments
        are invalid once FEs are absorbed; raises ``VcovTypeNotSupportedError``).
        Only available without absorbed fixed effects.

        Set ``cov_type="HAC"`` to use Newey-West (1987) heteroskedasticity-
        and autocorrelation-robust standard errors with a Bartlett kernel;
        the number of lags is given by *lags* and the time ordering by *time*.

        ``cluster`` takes precedence over *cov_type*: when *cluster* is
        supplied, cluster-robust standard errors are used regardless
        of the *cov_type* value.
    cluster : str or list of str, optional
        Column name(s) for cluster-robust standard errors.  Pass a single
        column name (e.g. ``cluster="firm"``).  Multi-way clustering is
        supported (pass a list of column names).
        Takes precedence over *cov_type*.
    lags : int, optional
        Number of lags for Newey-West HAC (required when ``cov_type="HAC"``).
    time : str, optional
        Column with the time index used to order observations for Newey-West
        HAC.  Observations are sorted by this column before fitting.
    hac_adjust : bool, default False
        Degrees-of-freedom correction for Newey-West HAC standard errors.
        When ``True``, the HAC variance is multiplied by ``N / (N - K)``,
        matching Stata's ``ivregress`` default behavior.
    entity : str, optional
        Column name for entity (panel unit) fixed effects. If provided,
        entity FE are absorbed via pyfixest's ``| f1 + f2`` syntax.
    time_fe : str, optional
        Column name for time fixed effects. If *entity* is also provided,
        two-way fixed effects are used.
    fixed_effects : list of str, optional
        Column names for arbitrary N-way fixed effects. Takes precedence
        over *entity*/*time_fe* when provided -- pass **either**
        ``entity=``/``time_fe=`` **or** ``fixed_effects=``, not both
        (raises ``ValueError``).

    Returns
    -------
    IVResult
        Immutable result object with coefficient arrays, weak-instrument
        diagnostics (Cragg-Donald Wald F-stat), and overidentification test
        (Hansen J statistic).

    Notes
    -----
    This function uses ``pyfixest.feols`` as the compute backend for
    non-HAC covariance types (HC1, iid, CRV1).  The Hansen J
    overidentification test and Cragg-Donald statistic are computed via
    a narrow ``linearmodels`` fallback call, since pyfixest does not
    expose these diagnostics.  ``linearmodels`` is already a project
    dependency (used by ``gmm()``, ``abond()``, ``PanelContext``).

    **Breaking changes** (v1.1):
    - The default ``cov_type`` changed from implicitly HC0 (via linearmodels
      ``"robust"`` + ``debiased=False``) to ``"HC1"`` (matching Stata
      ``ivregress 2sls, robust``).  This aligns with D1'.

    Examples
    --------
    >>> import open_econs as oe
    >>> r = oe.iv("y ~ w1 + w2 | x1 ~ z1 + z2", data=df)
    >>> r.tidy()
    >>> r.first_stage()
    >>> r.cragg_donald_stat
    """
    call = _capture_call(
        formula=formula, cov_type=cov_type, cluster=cluster, lags=lags,
        time=time, hac_adjust=hac_adjust,
        entity=entity, time_fe=time_fe, fixed_effects=fixed_effects,
    )

    cov_type = validate_cov_type(
        cov_type,
        accepted={"nonrobust", "HC0", "HC1", "HC2", "HC3", "robust",
                  "heteroskedastic", "unadjusted", "homoskedastic", "HAC",
                  "kernel"},
        estimator="iv()",
    )

    # ---- validate FE specification (same pattern as fe()) ----
    if fixed_effects is not None and (entity is not None or time_fe is not None):
        raise ValueError(
            "Pass either fixed_effects= OR entity=/time_fe=, not both. "
            "fixed_effects= takes precedence and ignores entity=/time_fe=."
        )

    # D2: forbid HC2/HC3 on models with absorbed fixed effects
    has_fe = (entity is not None or time_fe is not None or fixed_effects is not None)
    if cov_type in ("HC2", "HC3") and has_fe:
        raise VcovTypeNotSupportedError(cov_type)

    if cov_type == "clustered" and cluster is None:
        raise ValueError(
            "iv(): cov_type='clustered' requires a `cluster` column. "
            "Pass cluster='<column>' to request cluster-robust IV standard errors."
        )

    if cluster is not None:
        if isinstance(cluster, (list, tuple)):
            if len(cluster) > 1:
                raise NotImplementedError(
                    "iv(): multi-way clustering is not supported for IV-2SLS. "
                    "pyfixest CRV1 only supports one-way clustering for IV. "
                    "Pass a single cluster column (e.g. cluster='firm'). "
                    "For multi-way clustered IV, use linearmodels directly."
                )
            cluster = cluster[0]
        if cluster not in data.columns:
            raise errors.missing_column_error(cluster, data.columns.tolist())

    if cov_type == "HAC":
        if lags is None:
            raise ValueError(
                "Newey-West HAC requires `lags` (e.g. lags=1)."
            )
        if time is not None:
            data = data.sort_values(time)

    parsed = _parse_iv_formula(formula, data)
    has_inner_endog = parsed["has_inner_endog"]
    endog_vars = parsed["endog_vars"]
    exog_idx = parsed["exog_idx"]
    exog_vars_in_formula = [parsed["coef_names"][i] for i in exog_idx
                            if parsed["coef_names"][i] != "Intercept"]

    if not has_inner_endog:
        import warnings as _w
        _w.warn(
            "The legacy IV syntax 'y ~ rhs | instruments' treats ALL RHS variables "
            "as endogenous. This is almost certainly wrong if you have exogenous "
            "controls. Use the new syntax: 'y ~ exog | endog ~ instruments'. "
            "See the iv() docstring for details.",
            FutureWarning, stacklevel=3,
        )

    # Determine FE columns
    fe_parts: list[str] = []
    if fixed_effects is not None:
        fe_parts = list(fixed_effects)
    else:
        if entity is not None:
            fe_parts.append(entity)
        if time_fe is not None:
            fe_parts.append(time_fe)

    # ---- determine routing: pyfixest vs linearmodels fallback ----
    # linearmodels is used for: nonrobust, robust/HC1, HC0, HC2, HC3,
    #   cluster (debiased=False to match Stata ivregress default), HAC, kernel.
    # pyfixest is used for: FE models (entity/time_fe/fixed_effects) with
    #   HC1/robust vcov.  FE absorption is the primary pyfixest capability
    #   for IV; linearmodels doesn't support FE in IV.
    # Source for cluster exception: ivregress.ado lines 637-714 — no SSC
    # applied without `small` option.
    use_lm = not has_fe  # linearmodels for all non-FE cases

    if use_lm:
        return _iv_linearmodels_path(
            formula=formula,
            parsed=parsed,
            cov_type=cov_type,
            cluster=cluster,
            lags=lags,
            time=time,
            hac_adjust=hac_adjust,
            fe_parts=fe_parts,
            exog_vars=exog_vars_in_formula,
            endog_vars=endog_vars,
            call=call,
        )

    # ---- pyfixest path ----
    return _iv_pyfixest_path(
        formula=formula,
        parsed=parsed,
        cov_type=cov_type,
        cluster=cluster,
        lags=lags,
        time=time,
        hac_adjust=hac_adjust,
        fe_parts=fe_parts,
        exog_vars=exog_vars_in_formula,
        endog_vars=endog_vars,
        call=call,
    )


def _iv_pyfixest_path(
    *,
    formula: str,
    parsed: dict,
    cov_type: str,
    cluster: str | list[str] | None,
    lags: int | None,
    time: str | None,
    hac_adjust: bool,
    fe_parts: list[str],
    exog_vars: list[str],
    endog_vars: list[str],
    call: dict[str, Any],
) -> IVResult:
    """Run IV estimation through pyfixest and translate to IVResult."""
    import pyfixest as pf

    data_index = parsed["index"]
    y_arr = parsed["y"]
    X_full = parsed["X"]
    all_cols = parsed["coef_names"]
    endog_idx = parsed["endog_idx"]
    instr_matrix = parsed["instr_matrix"]

    # Build working dataframe: LHS + RHS + FE + cluster + instruments + time
    dep_var = formula.split("~")[0].strip()
    y_name = dep_var
    endog_col_names = [all_cols[i] for i in endog_idx]
    exog_col_names = [c for c in all_cols if c not in endog_col_names]

    # Collect all needed columns
    needed_cols = list(all_cols)
    needed_cols.extend(fe_parts)
    if cluster is not None:
        needed_cols.append(cluster)
    if cov_type == "HAC" and time is not None:
        needed_cols.append(time)
    # Instrument columns
    instr_formula = formula.split("|", 1)[1].strip()
    if "~" in instr_formula:
        instr_expr = instr_formula.split("~", 1)[1].strip()
    else:
        instr_expr = instr_formula
    from open_econs.models.linear.iv import _extract_vars as _ev
    instr_col_names = _ev(instr_expr)
    needed_cols.extend(c for c in instr_col_names if c not in needed_cols)

    # Deduplicate
    seen: set[str] = set()
    unique_cols: list[str] = []
    for c in needed_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)

    # We need the original data for FE/cluster/time columns
    # Since we don't have it directly, reconstruct from parsed
    # Actually, we need the original data passed to iv(). Let's store it in parsed.
    # For now, we'll need to reconstruct from the formula parsing.
    # The simplest approach: parse the IV formula for pyfixest directly.

    # Build the pyfixest formula:
    # Without FE: "y ~ exog | endog ~ instruments"
    # With FE: "y ~ exog | fe1 + fe2 | endog ~ instruments"
    exog_part = " + ".join(c for c in exog_col_names if c != "Intercept")
    endog_part = " + ".join(endog_col_names)
    instr_part = " + ".join(instr_col_names)

    if exog_part:
        pf_y_rhs = f"{y_name} ~ {exog_part}"
    else:
        pf_y_rhs = f"{y_name} ~ 1"

    iv_block = f"{endog_part} ~ {instr_part}"

    if fe_parts:
        fe_block = " + ".join(fe_parts)
        pf_fml = f"{pf_y_rhs} | {fe_block} | {iv_block}"
    else:
        pf_fml = f"{pf_y_rhs} | {iv_block}"

    # Determine vcov for pyfixest
    pf_vcov: Any
    cov_label: str
    if cluster is not None:
        pf_vcov = {"CRV1": cluster}
        cov_label = f"clustered({cluster})"
    elif cov_type == "HAC":
        # pyfixest NW requires post-hoc vcov() call; fit with HC1 first
        pf_vcov = "HC1"
        cov_label = f"HAC({lags})"
    elif cov_type in ("nonrobust", "unadjusted", "homoskedastic"):
        pf_vcov = "iid"
        cov_label = cov_type
    elif cov_type in ("HC1", "robust", "heteroskedastic"):
        pf_vcov = "HC1"
        cov_label = "robust"
    else:
        pf_vcov = "HC1"
        cov_label = cov_type

    # Build working dataframe
    work_data_cols = [y_name] + [c for c in all_cols if c != "Intercept"]
    work_data_cols.extend(fe_parts)
    if cluster is not None:
        work_data_cols.append(cluster)
    if cov_type == "HAC" and time is not None:
        work_data_cols.append(time)
    work_data_cols = list(dict.fromkeys(work_data_cols))  # deduplicate preserving order

    # We need the original data to get FE/cluster/time columns
    original_data = parsed.get("original_data")

    # For now, work with the data we have: we know the index alignment
    # The y_arr and X_full are aligned to data_index
    work = pd.DataFrame({y_name: y_arr}, index=data_index)
    for i, col in enumerate(all_cols):
        if col == "Intercept":
            continue
        xi = X_full[:, [j for j, c in enumerate(all_cols) if c == col]]
        if xi.shape[1] > 0:
            work[col] = xi.ravel()

    # Add instrument columns from the original data
    for i, col in enumerate(instr_col_names):
        if col not in work.columns:
            work[col] = instr_matrix[:, i] if instr_matrix.shape[1] > i else 0.0

    # Add FE, cluster, and time columns from the original data
    if original_data is not None:
        extra_cols = list(fe_parts)
        if cluster is not None:
            if isinstance(cluster, list):
                extra_cols.extend(cluster)
            else:
                extra_cols.append(cluster)
        if cov_type == "HAC" and time is not None:
            extra_cols.append(time)
        for col in extra_cols:
            if col not in work.columns and col in original_data.columns:
                work[col] = original_data.loc[data_index, col].values

    # Validate that FE columns are present in the working data
    for col in fe_parts:
        if col not in work.columns:
            raise ValueError(
                f"iv(): FE column '{col}' not found in working data. "
                "This indicates an internal error in formula parsing."
            )

    # Fit with pyfixest
    fit = pf.feols(pf_fml, data=work, vcov=pf_vcov)

    # Apply HAC post-hoc if needed
    # NOTE: pyfixest's NW already applies N/(N-K) internally (matching
    # Stata's `newey`).  The `hac_adjust` parameter is a no-op here;
    # it only affects the linearmodels fallback path.
    if cov_type == "HAC" and time is not None and cluster is None:
        fit.vcov("NW", vcov_kwargs={"time_id": time, "lag": lags})

    # Extract results from pyfixest
    kept_columns = list(fit.coef().index)
    coef_dict = fit.coef().to_dict()
    se_dict = fit.se().to_dict()
    tstat_dict = fit.tstat().to_dict()
    pvalue_dict = fit.pvalue().to_dict()
    ci_df = fit.confint()

    coef_arr = np.array([coef_dict.get(c, 0.0) for c in kept_columns])
    se_arr = np.array([se_dict.get(c, np.nan) for c in kept_columns])
    t_arr = np.array([tstat_dict.get(c, np.nan) for c in kept_columns])
    p_arr = np.array([pvalue_dict.get(c, np.nan) for c in kept_columns])

    conf_lower = np.array([ci_df.loc[c, ci_df.columns[0]] if c in ci_df.index else np.nan for c in kept_columns])
    conf_upper = np.array([ci_df.loc[c, ci_df.columns[1]] if c in ci_df.index else np.nan for c in kept_columns])
    conf_arr = np.column_stack([conf_lower, conf_upper])

    n = int(fit._N)
    k = len(kept_columns)

    # Compute absorbed DOF
    n_absorbed = _count_absorbed_dof(work, fe_parts) if fe_parts else 0
    df_resid_adj = max(n - n_absorbed - k, 1)

    # Residuals and fitted values
    residuals_arr = fit.resid()
    fitted_arr = y_arr[:len(residuals_arr)] - residuals_arr

    # First-stage F statistics
    fs_f_stats = {}
    first_stage_model = fit._model_1st_stage
    if first_stage_model is not None:
        # Joint first-stage F for all instruments
        joint_f = float(fit._f_stat_1st_stage)
        # For single endogenous variable, this IS the per-endog F
        for ev in endog_vars:
            fs_f_stats[ev] = joint_f

    fs_f_series = pd.Series(fs_f_stats, name="F") if fs_f_stats else pd.Series(dtype=float, name="F")

    # Cragg-Donald: min of per-endogenous first-stage Fs
    try:
        cragg_donald = float(min(fs_f_stats.values())) if fs_f_stats else float("nan")
    except Exception:
        cragg_donald = float("nan")

    # Hansen J: requires linearmodels fallback (pyfixest doesn't expose it)
    hansen_j, hansen_p = _compute_hansen_j(
        y_arr=parsed["y"],
        X_full=X_full,
        exog_idx=parsed["exog_idx"],
        endog_idx=parsed["endog_idx"],
        instr_matrix=instr_matrix,
        index=parsed["index"],
        data_index=data_index,
    )

    # Covariance matrix
    _cov = pd.DataFrame(
        fit._vcov,
        index=kept_columns,
        columns=kept_columns,
    )

    # RSd
    ssr = float(np.sum(residuals_arr ** 2))
    rsd = float(np.sqrt(ssr / max(df_resid_adj, 1)))

    result = IVResult(
        formula=formula,
        nobs=n,
        df_resid=df_resid_adj,
        df_model=k,
        cov_type=cov_label,
        coefficients=pd.Series(coef_arr, index=kept_columns),
        std_errors=pd.Series(se_arr, index=kept_columns),
        z_stats=pd.Series(t_arr, index=kept_columns),
        p_values=pd.Series(p_arr, index=kept_columns),
        conf_int=pd.DataFrame({"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]}, index=kept_columns),
        rsd=rsd,
        first_stage_f=fs_f_series,
        cragg_donald_f=cragg_donald,
        hansen_j_stat=hansen_j,
        hansen_j_p=hansen_p,
        fitted=pd.Series(fitted_arr[:len(data_index)], index=data_index[:len(fitted_arr)], name="fitted"),
        residuals=pd.Series(residuals_arr[:len(data_index)], index=data_index[:len(residuals_arr)], name="residuals"),
        call=call,
        _fit=None,
        _exog_names=exog_vars,
        _endog_names=endog_vars,
    )
    object.__setattr__(result, "_cov", _cov)
    return result


def _iv_linearmodels_path(
    *,
    formula: str,
    parsed: dict,
    cov_type: str,
    cluster: str | list[str] | None,
    lags: int | None,
    time: str | None,
    hac_adjust: bool,
    fe_parts: list[str],
    endog_vars: list[str],
    exog_vars: list[str],
    call: dict[str, Any],
) -> IVResult:
    """Fallback path using linearmodels for HC0/HC2/HC3 and cluster.

    For cluster-robust IV, this route is used because Stata's ``ivregress 2sls``
    default (no ``small`` option) applies **no SSC** to the cluster VCE
    (``ivregress.ado`` lines 637-714).  pyfixest's CRV1 applies
    ``(N-1)/(N-K) * G/(G-1)`` unconditionally, which matches Stata's ``small``
    variant, not the default.  ``linearmodels(debiased=False)`` matches
    Stata's default exactly.
    """
    from typing import cast as _cast

    y_arr = parsed["y"]
    X_full = parsed["X"]
    exog_idx = parsed["exog_idx"]
    endog_idx = parsed["endog_idx"]
    instr_matrix = parsed["instr_matrix"]

    _IV_COV_MAP = {
        "nonrobust": "unadjusted",
        "HC0": "robust",
        "HC1": "robust",
        "HC2": "robust",
        "HC3": "robust",
        "robust": "robust",
        "heteroskedastic": "robust",
        "unadjusted": "unadjusted",
        "homoskedastic": "unadjusted",
    }
    _IV_DEBIAS_MAP = {
        "nonrobust": False,
        "HC0": False,
        "HC1": True,
        "HC2": False,
        "HC3": False,
        "robust": False,
        "unadjusted": False,
        "homoskedastic": False,
    }

    Z_arr = instr_matrix if instr_matrix.shape[1] > 0 else None
    X_exog = X_full[:, exog_idx] if exog_idx else None
    X_endog = X_full[:, endog_idx] if endog_idx else None

    # Get cluster array from original data if needed
    cluster_arr = None
    if cluster is not None:
        original_data = parsed.get("original_data")
        if original_data is not None and cluster in original_data.columns:
            cluster_arr = original_data.loc[parsed["index"], cluster].values
        else:
            raise ValueError(
                f"iv(): cluster column '{cluster}' not found in data."
            )

    from linearmodels.iv import IV2SLS as LM_IV2SLS
    try:
        if cluster is not None:
            # Route through linearmodels with debiased=False to match
            # Stata's ivregress 2sls default (no SSC).
            # Source: ivregress.ado lines 637-714.
            fitted = LM_IV2SLS(y_arr, X_exog, X_endog, Z_arr).fit(
                cov_type="clustered",
                clusters=cluster_arr,
                debiased=False,
            )
            cov_label = f"clustered({cluster})"
        elif cov_type == "HAC":
            fitted = LM_IV2SLS(y_arr, X_exog, X_endog, Z_arr).fit(
                cov_type="kernel",
                debiased=hac_adjust,
                bandwidth=lags,
            )
            cov_label = f"HAC({lags})"
        else:
            fitted = LM_IV2SLS(y_arr, X_exog, X_endog, Z_arr).fit(
                cov_type=_IV_COV_MAP.get(cov_type, "robust"),
                debiased=_IV_DEBIAS_MAP.get(cov_type, False),
            )
            cov_label = cov_type
    except Exception as e:
        raise RuntimeError(f"IV2SLS estimation failed: {e}") from e

    coef_arr = fitted.params.values
    se_arr = fitted.std_errors.values
    z_arr = fitted.tstats.values
    p_arr = fitted.pvalues.values
    conf_arr = fitted.conf_int(level=0.95).values

    coef_names = fitted.params.index.tolist()
    conf_int = pd.DataFrame(
        {"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]},
        index=coef_names,
    )

    residuals_arr = fitted.resids.values
    fitted_arr = y_arr - residuals_arr

    # First-stage F
    fs_f_stats = {}
    for en_name in fitted.model.endog.cols:
        fs = _cast(Any, fitted).first_stage
        if fs is not None and en_name in fs.individual:
            ind_res = fs.individual[en_name]
            f_stat = ind_res.f_statistic.stat if hasattr(ind_res, "f_statistic") else float("nan")
            fs_f_stats[en_name] = float(f_stat)
        else:
            fs_f_stats[en_name] = float("nan")

    fs_f_series = pd.Series(fs_f_stats, name="F")

    try:
        cragg_donald = float(min(fs_f_stats.values())) if fs_f_stats else float("nan")
    except Exception:
        cragg_donald = float("nan")

    # Hansen J
    try:
        overid = _cast(Any, fitted).sargan
        hansen_j = float(overid.stat)
        hansen_p = float(overid.pval)
    except (AttributeError, Exception):
        hansen_j = float("nan")
        hansen_p = float("nan")

    df_resid_adj = max(int(fitted.df_resid), 1)

    return IVResult(
        formula=formula,
        nobs=int(fitted.nobs),
        df_resid=df_resid_adj,
        df_model=int(fitted.df_model),
        cov_type=cov_label,
        coefficients=pd.Series(coef_arr, index=coef_names),
        std_errors=pd.Series(se_arr, index=coef_names),
        z_stats=pd.Series(z_arr, index=coef_names),
        p_values=pd.Series(p_arr, index=coef_names),
        conf_int=conf_int,
        rsd=float(np.sqrt(fitted.s2)) if hasattr(fitted, "s2") else float("nan"),
        first_stage_f=fs_f_series,
        cragg_donald_f=cragg_donald,
        hansen_j_stat=hansen_j,
        hansen_j_p=hansen_p,
        fitted=pd.Series(fitted_arr, index=parsed["index"], name="fitted"),
        residuals=pd.Series(residuals_arr, index=parsed["index"], name="residuals"),
        call=call,
        _fit=fitted,
        _exog_names=exog_vars,
        _endog_names=endog_vars,
    )


def _compute_hansen_j(
    *,
    y_arr: np.ndarray,
    X_full: np.ndarray,
    exog_idx: list[int],
    endog_idx: list[int],
    instr_matrix: np.ndarray,
    index: pd.Index,
    data_index: pd.Index,
) -> tuple[float, float]:
    """Compute Hansen J overidentification test via linearmodels.

    pyfixest does not expose Hansen J / Sargan for IV models.
    This narrow fallback calls linearmodels solely for the overid diagnostic.
    linearmodels is already a project dependency (gmm, abond, PanelContext).
    """
    from typing import cast as _cast

    Z_arr = instr_matrix if instr_matrix.shape[1] > 0 else None
    X_exog = X_full[:, exog_idx] if exog_idx else None
    X_endog = X_full[:, endog_idx] if endog_idx else None

    # Check if we have enough instruments for overidentification
    n_endog = len(endog_idx)
    n_instr = instr_matrix.shape[1] if instr_matrix.ndim == 2 else 0
    if n_instr <= n_endog:
        # Just-identified: Hansen J is exactly 0
        return 0.0, 1.0

    from linearmodels.iv import IV2SLS as LM_IV2SLS
    try:
        lm_fit = LM_IV2SLS(y_arr, X_exog, X_endog, Z_arr).fit()
        overid = _cast(Any, lm_fit).sargan
        return float(overid.stat), float(overid.pval)
    except Exception:
        return float("nan"), float("nan")


def _extract_vars(expr: str) -> list[str]:
    import re
    terms = re.split(r"[+\-*/]", expr)
    result = []
    for t in terms:
        t = t.strip()
        if t and t not in ("1", "0", "-1"):
            result.append(t)
    return result


def _parse_iv_formula(formula: str, data: pd.DataFrame) -> dict:
    """Parse an IV/2SLS/GMM formula into aligned model matrices.

    Shared by :func:`iv` and :func:`open_econs.models.linear.gmm.gmm` so the
    ``y ~ exog | endog ~ instruments`` grammar is implemented exactly once.

    Returns a dict with:

    * ``y`` : ndarray (n,) dependent vector.
    * ``X`` : ndarray (n, p) full regressor matrix (incl. intercept).
    * ``coef_names`` : list[str] of length ``p`` (X column labels).
    * ``exog_idx`` : positions of exogenous regressors (incl. intercept) in ``X``.
    * ``endog_idx`` : positions of endogenous regressors in ``X``.
    * ``instr_matrix`` : ndarray (n, L_instr) instrument variables only (intercept
      dropped) -- the explicit instruments named after ``~`` in the inner block.
    * ``has_inner_endog`` : whether the ``| endog ~ instruments`` syntax was used.
    * ``endog_vars`` : list of endogenous variable names.
    * ``index`` : pandas.Index of the kept (aligned, non-missing) rows.
    * ``dropped`` / ``original_n`` : row accounting.

    The instrument matrix ``Z`` handed to a GMM solver must be the exogenous
    regressors (their own instruments) concatenated with ``instr_matrix``;
    the caller assembles that.  Rows of ``y``, ``X`` and ``instr_matrix`` are
    aligned to a common index so they never silently disagree.
    """
    if "|" not in formula:
        raise ValueError(
            "This estimator requires a three-part formula: 'y ~ exog | endog ~ instruments'"
        )

    y_part, instr_part = formula.split("|", 1)
    y_part = y_part.strip()
    instr_part = instr_part.strip()

    rhs_split = y_part.split("~", 1)
    dep_var = rhs_split[0].strip()
    rhs_expr = rhs_split[1].strip() if len(rhs_split) > 1 else ""

    has_inner_endog = "~" in instr_part
    if has_inner_endog:
        endog_part, instr_expr = instr_part.split("~", 1)
        endog_expr = endog_part.strip()
        instr_expr = instr_expr.strip()
        endog_vars = _extract_vars(endog_expr)
        exog_vars = _extract_vars(rhs_expr)
        all_rhs_vars = list(dict.fromkeys(exog_vars + endog_vars))
        full_rhs_formula = " + ".join(all_rhs_vars)
        y_formula = f"{dep_var} ~ {full_rhs_formula}"
    else:
        all_rhs_vars = _extract_vars(rhs_expr)
        endog_vars = all_rhs_vars
        exog_vars = []
        instr_expr = instr_part
        full_rhs_formula = " + ".join(all_rhs_vars)
        y_formula = f"{dep_var} ~ {full_rhs_formula}"

    from formulaic import Formula

    try:
        matrices = Formula(y_formula).get_model_matrix(data, na_action="drop")
    except Exception as e:
        msg = str(e)
        if "not present in the dataset" in msg or "is not present" in msg:
            import re as _re
            m = _re.search(r"`(\w+)`", msg)
            bad_col = m.group(1) if m else y_part
            raise errors.missing_column_error(bad_col, data.columns.tolist()) from e
        raise

    if hasattr(matrices, "rhs"):
        XX = matrices.rhs
        yy = matrices.lhs
    else:
        from open_econs._internal.formula import parse_formula as _parse
        yy, XX = _parse(y_formula, data)
        matrices = None

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

    x_index = XX.index
    instr_matrices = Formula(instr_expr).get_model_matrix(
        data.loc[x_index], na_action="drop",
    )
    # A single-sided instrument formula ("z1 + z2") returns a bare DataFrame;
    # a two-sided one returns a ModelMatrix with .rhs/.lhs.
    Z_instr = instr_matrices.rhs if hasattr(instr_matrices, "rhs") else instr_matrices

    # Align instrument rows to the regressor rows (instruments may carry their
    # own missing values), so y / X / instruments are always the same length.
    keep = x_index.intersection(Z_instr.index)
    yy = yy.loc[keep]
    XX = XX.loc[keep]
    Z_instr = Z_instr.loc[keep]

    y_arr = yy.values.ravel().astype(float)
    X_full = XX.values.astype(float)
    all_cols = XX.columns.tolist()
    instr_cols = [c for c in Z_instr.columns if c != "Intercept"]
    instr_matrix = (
        Z_instr[instr_cols].values.astype(float) if instr_cols
        else np.zeros((len(X_full), 0))
    )

    endog_cols_in_model = [c for c in all_cols if c in endog_vars]
    exog_idx = [i for i, c in enumerate(all_cols) if c not in endog_cols_in_model]
    endog_idx = [i for i, c in enumerate(all_cols) if c in endog_cols_in_model]

    return {
        "y": y_arr,
        "X": X_full,
        "coef_names": all_cols,
        "exog_idx": exog_idx,
        "endog_idx": endog_idx,
        "instr_matrix": instr_matrix,
        "has_inner_endog": has_inner_endog,
        "endog_vars": endog_vars,
        "index": keep,
        "dropped": original_n - len(keep),
        "original_n": original_n,
        "original_data": data,
    }


def _count_absorbed_dof(df: pd.DataFrame, fe_cols: list[str]) -> int:
    """Count absorbed degrees of freedom for N-way FE.

    Same formula as fe.py: ``sum(n_groups_i) - (k - 1)`` where ``k = len(fe_cols)``.
    """
    if not fe_cols:
        return 0
    n_groups = sum(df[c].nunique() for c in fe_cols if c in df.columns)
    return n_groups - (len(fe_cols) - 1)
