from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from open_econs.core.call_capture import capture_call as _capture_call
from open_econs._internal import errors
from open_econs.core.results import CountResult


def poisson(
    formula: str,
    data: pd.DataFrame,
    fixed_effects: list[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
    cluster: str | list[str] | None = None,
    cov_type: str = "HC1",
    vcov_backend: str = "fixest",
    offset: str | None = None,
    weights: str | None = None,
    separation_check: list[str] | None = None,
) -> CountResult:
    """Estimate a Poisson pseudo-maximum-likelihood (PPML) fixed-effects model.

    Wraps ``pyfixest.fepois`` — the Python port of R's ``fixest::fepois`` — and
    is reconciled to Stata SSC ``ppmlhdfe`` (Correia, Guimaraes & Zylkin 2020)
    and R ``fixest::fepois`` (Berge 2018) to a numeric tolerance of ``1e-6``.
    Fixed effects are absorbed via the alternating-projections IRLS core (the
    same demeaner used by :func:`open_econs.fe`). This is the standard estimator
    for count outcomes and for multiplicative / gravity models (Santos Silva &
    Tenreyro 2006), consistent even when the outcome is continuous non-negative.

    Parameters
    ----------
    formula : str
        Two-sided formula, e.g. ``"y ~ x1 + x2"``. Do **not** put the
        fixed-effect indicators in the formula; use *fixed_effects* or
        *entity*/*time*.
    data : pd.DataFrame
        Data with all formula variables plus FE / cluster / offset columns.
    fixed_effects : list of str, optional
        Column names of N-way fixed effects to absorb. Takes precedence over
        *entity*/*time*; pass **either** ``fixed_effects=`` **or**
        ``entity=``/``time=``, not both.
    entity, time : str, optional
        Convenience two-way FE shorthand (mapped to ``fixed_effects``).
    cluster : str or list of str, optional
        Column(s) for cluster-robust (CRV1) standard errors; a list requests
        multi-way clustering. Takes precedence over *cov_type*.
    cov_type : str, default "HC1"
        Heteroskedasticity-robust estimator used when *cluster* is not set.
        One of ``"nonrobust"``, ``"HC1"``. (Poisson PML is robust; ``"HC1"`` is
        the sandwich/robust meat, matching ``ppmlhdfe`` / ``fepois`` defaults.)
    vcov_backend : {"fixest", "stata"}, default "fixest"
        Small-sample convention for the reported variance (rule-15 toggle).
        Only rescales the cluster/robust variance — point estimates, deviance,
        and log-likelihood are identical.

        - ``"fixest"`` (default): matches R ``fixest::fepois`` /
          ``pyfixest.fepois`` defaults (``k_adj=True, G_adj=True``).
        - ``"stata"``: matches Stata ``ppmlhdfe`` via
          ``ssc(k_adj=False, G_adj=True, k_fixef="none")``.

        See ``methodology/limited/poisson.md`` for the derivation.
    offset : str, optional
        Column name of an exposure offset entered with coefficient fixed at 1
        (log-exposure). Maps to ``pyfixest.fepois(offset=...)``.
    weights : str, optional
        Column name of frequency/analytic weights.
    separation_check : list of str, optional
        Separation-detection methods passed to ``pyfixest.fepois`` (e.g.
        ``["fe"]``, ``["ir"]``). ``None`` uses the pyfixest default.

    Returns
    -------
    CountResult
        Immutable result. Coefficients are on the log (index) scale. Use
        ``.irr()`` for incidence-rate ratios, ``.margins()`` for average
        marginal effects on the count scale, ``.predict()`` for fitted means.

    Examples
    --------
    >>> import open_econs as oe
    >>> r = oe.poisson("y ~ x1 + x2", data=df, fixed_effects=["firm", "year"], cluster="firm")
    >>> r.tidy(); r.irr(); r.margins()
    >>> r_stata = oe.poisson("y ~ x1 + x2", data=df, fixed_effects=["firm", "year"],
    ...                      cluster="firm", vcov_backend="stata")
    """
    call = _capture_call(
        formula=formula, fixed_effects=fixed_effects, entity=entity, time=time,
        cluster=cluster, cov_type=cov_type, vcov_backend=vcov_backend,
        offset=offset, weights=weights, separation_check=separation_check,
    )

    if vcov_backend not in ("fixest", "stata"):
        raise ValueError(
            f"vcov_backend must be 'fixest' or 'stata', got {vcov_backend!r}."
        )
    if cov_type not in ("nonrobust", "HC1"):
        raise ValueError(
            f"poisson() supports cov_type in {{'nonrobust', 'HC1'}}, got "
            f"{cov_type!r}. Use cluster=... for cluster-robust SEs."
        )

    # ---- validate FE specification ----
    if fixed_effects is not None and (entity is not None or time is not None):
        raise ValueError(
            "Pass either fixed_effects= OR entity=/time=, not both. "
            "fixed_effects= takes precedence and ignores entity=/time=."
        )
    fe_parts: list[str] = []
    if fixed_effects is not None:
        fe_parts = list(fixed_effects)
    else:
        if entity is not None:
            fe_parts.append(entity)
        if time is not None:
            fe_parts.append(time)
    if not fe_parts:
        raise ValueError(
            "poisson() requires fixed effects: pass fixed_effects= or "
            "entity=/time=. (Pooled Poisson without FE is out of scope.)"
        )

    # ---- resolve column names from the formula ----
    lhs = formula.split("~", 1)[0].strip()
    rhs_formula = formula.split("~", 1)[1].strip()
    x_terms = [t.strip() for t in rhs_formula.split("+") if t.strip() and t.strip() != "1"]

    needed = [lhs, *x_terms, *fe_parts]
    if isinstance(cluster, str):
        needed.append(cluster)
    elif isinstance(cluster, list):
        needed.extend(cluster)
    if offset is not None:
        needed.append(offset)
    if weights is not None:
        needed.append(weights)
    missing = [c for c in dict.fromkeys(needed) if c not in data.columns]
    if missing:
        raise errors.missing_column_error(missing[0], data.columns.tolist())

    # ---- drop rows with NA in used columns ----
    used_cols = list(dict.fromkeys(needed))
    original_n = len(data)
    work = data[used_cols].dropna()
    dropped = original_n - len(work)
    if dropped > 0:
        import warnings as _w
        _w.warn(
            errors.rows_dropped_warning(dropped, original_n, []),
            RuntimeWarning,
            stacklevel=3,
        )
    if len(work) == 0:
        raise errors.empty_data_error(original_n, dropped, [])

    # ---- build pyfixest call ----
    import pyfixest as pf

    fe_suffix = " + ".join(fe_parts)
    x_part = " + ".join(x_terms)
    pf_fml = f"{lhs} ~ {x_part} | {fe_suffix}"

    pf_vcov: Any
    if cluster is not None:
        cl = cluster if isinstance(cluster, str) else " + ".join(cluster)
        pf_vcov = {"CRV1": cl}
        cov_label = f"cluster({cluster})" if isinstance(cluster, str) else "cluster(" + ", ".join(cluster) + ")"
    else:
        pf_vcov = "iid" if cov_type == "nonrobust" else "hetero"
        cov_label = cov_type

    if vcov_backend == "stata":
        pf_ssc = pf.ssc(k_adj=False, G_adj=True, k_fixef="none")
    else:
        pf_ssc = pf.ssc()

    fit = pf.fepois(
        pf_fml,
        data=work,
        vcov=pf_vcov,
        offset=offset,
        weights=weights,
        ssc=pf_ssc,
        separation_check=separation_check,
    )

    # ---- extract ----
    coef_dict = fit.coef().to_dict()
    se_dict = fit.se().to_dict()
    tstat_dict = fit.tstat().to_dict()
    pvalue_dict = fit.pvalue().to_dict()
    ci_df = fit.confint()

    cols = [str(c) for c in x_terms]

    coef_arr = np.array([coef_dict.get(c, np.nan) for c in cols])
    se_arr = np.array([se_dict.get(c, np.nan) for c in cols])
    z_arr = np.array([tstat_dict.get(c, np.nan) for c in cols])
    p_arr = np.array([pvalue_dict.get(c, np.nan) for c in cols])
    conf_lower = np.array([ci_df.loc[c, ci_df.columns[0]] if c in ci_df.index else np.nan for c in cols])
    conf_upper = np.array([ci_df.loc[c, ci_df.columns[1]] if c in ci_df.index else np.nan for c in cols])

    # ---- ppmlhdfe non-clustered robust (CGZ) bread ----
    # When matching Stata ppmlhdfe with no cluster(), ppmlhdfe reports a
    # *robust (sandwich)* SE, not an OIM iid SE. Its robust meat uses the
    # Correia-Guimaraes-Zylkin (2019) nonlinear-Poisson adjustment
    #   meat = Σ_i (y_i − μ_i)^2 / μ_i · x_i x_i'
    # with the OIM bread (X'WX)^{-1} and the small-sample k_adj = (N−1)/(N−K)
    # factor. fixest/pyfixest "hetero" does NOT apply the 1/μ nonlinearity
    # scaling, so it diverges from ppmlhdfe by ~4e-4. We wrap ppmlhdfe's exact
    # meat here (rule 15 option a) so vcov_backend="stata" reproduces it to
    # ≤1e-6. The fixest backend (default) is untouched. See
    # methodology/limited/poisson.md §2.4.
    if cluster is None and vcov_backend == "stata":
        V = _ppmlhdfe_robust_vcov(fit, work, lhs, cols)
        se_arr = np.sqrt(np.diag(V))
        from scipy import stats as _st
        z_arr = coef_arr / se_arr
        p_arr = 2.0 * (1.0 - _st.norm.cdf(np.abs(z_arr)))
        z_crit = _st.norm.ppf(0.975)
        conf_lower = coef_arr - z_crit * se_arr
        conf_upper = coef_arr + z_crit * se_arr
        _cov = pd.DataFrame(V, index=cols, columns=cols)
        cov_label = "robust"

    n = int(fit._N)
    k = len(cols)
    n_absorbed = _count_absorbed_dof(work, fe_parts)
    df_resid = max(n - n_absorbed - k, 1)

    if not (cluster is None and vcov_backend == "stata"):
        _cov = pd.DataFrame(fit._vcov, index=cols, columns=cols)

    mu = np.asarray(fit.predict(type="response"))
    fitted_values = pd.Series(mu, index=work.index, name="fitted_mean")

    llf = float(getattr(fit, "_loglik", float("nan")))
    deviance = float(getattr(fit, "deviance", float("nan")))
    pseudo_r2 = float(getattr(fit, "_pseudo_r2", float("nan")))

    result = CountResult(
        formula=formula,
        rhs_formula=rhs_formula,
        nobs=n,
        df_resid=df_resid,
        df_model=k,
        cov_type=cov_label,
        coefficients=pd.Series(coef_arr, index=cols),
        std_errors=pd.Series(se_arr, index=cols),
        z_stats=pd.Series(z_arr, index=cols),
        p_values=pd.Series(p_arr, index=cols),
        conf_int=pd.DataFrame({"lower": conf_lower, "upper": conf_upper}, index=cols),
        llf=llf,
        deviance=deviance,
        pseudo_r2=pseudo_r2,
        n_absorbed=n_absorbed,
        fixed_effects=fe_parts,
        fitted=fitted_values,
        call=call,
        vcov_backend=vcov_backend,
        _cov=_cov,
        _fit=fit,
    )
    return result


def _count_absorbed_dof(df: pd.DataFrame, fe_cols: list[str]) -> int:
    """Absorbed degrees of freedom for N-way FE: sum(n_groups) - (k - 1)."""
    if not fe_cols:
        return 0
    n_groups = sum(df[c].nunique() for c in fe_cols)
    return n_groups - (len(fe_cols) - 1)


def _ppmlhdfe_robust_vcov(fit: Any, work: pd.DataFrame, lhs: str, cols: list[str]) -> np.ndarray:
    """ppmlhdfe non-clustered robust (sandwich) variance for Poisson PML.

    Wraps Stata ``ppmlhdfe``'s exact robust bread (Correia-Guimaraes-Zylkin
    2019 nonlinearity adjustment) so ``oe.poisson(vcov_backend="stata")`` with
    no ``cluster=`` matches ``ppmlhdfe``'s non-clustered SE to ≤1e-6.

    The robust meat scales each score contribution by ``1/μ_i``:

        meat = Σ_i (y_i − μ_i)^2 / μ_i · x_i x_i'

    combined with the OIM bread ``(X'WX)^{-1}`` (``fit._bread``, ``W =
    diag(μ)``) and the small-sample ``k_adj = (N−1)/(N−K)`` factor that
    ppmlhdfe applies to its non-clustered robust SE. ``fit._X`` is the
    fixed-effects-residualized regressor matrix, and ``mu`` is the fitted
    conditional mean on the original outcome scale (``fit.predict(type=
    "response")``). ``y`` is taken from *work* and aligned to the estimation
    sample by dropping the row labels recorded in ``fit.na_index`` (NA and
    separated observations).
    """
    X = np.asarray(fit._X, dtype=float)
    mu = np.asarray(fit.predict(type="response"), dtype=float).ravel()
    mu = np.maximum(mu, np.finfo(float).tiny)

    na = np.asarray(fit.na_index)
    ys = work[lhs]
    if na.size:
        ys = ys.drop(na.astype(int))
    y = np.asarray(ys.values, dtype=float).ravel()

    bread = np.asarray(fit._bread, dtype=float)
    resid = y - mu
    meat = ((resid ** 2) / mu)[:, None, None] * (X[:, :, None] * X[:, None, :])
    meat = meat.sum(axis=0)

    N = y.shape[0]
    K = len(cols)
    k_adj = (N - 1) / (N - K)
    return k_adj * (bread @ meat @ bread)
