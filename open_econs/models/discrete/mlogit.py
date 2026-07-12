
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm as _norm
from typing import Any

from open_econs.core.call_capture import capture_call as _capture_call
from open_econs._internal import errors
from open_econs.core.results import MultinomialResult
from open_econs.core.cov_type import validate_cov_type


def mlogit(
    formula: str,
    data: pd.DataFrame,
    *,
    base: object = None,
    cov_type: str = "HC2",
    cluster: str | None = None,
    **kwargs: Any,
) -> MultinomialResult:
    """Estimate a multinomial logit (MNLogit) model.

    Parameters
    ----------
    formula : str
        Two-sided formula string, e.g. ``"y ~ x1 + x2"``.  The left-hand side is
        the categorical outcome; it is treated as a single label vector (not
        dummy-expanded).
    data : pd.DataFrame
        Data containing all variables referenced in *formula*.
    base : object, default None
        Baseline (reference) outcome category.  Defaults to **``None``**, which
        picks the **first category in sorted/alphabetical order** of the
        outcome's distinct values.  This matches statsmodels' and R's native
        default and is chosen because it is deterministic and does not depend on
        sample frequencies.

        .. warning::
           This deliberately disagrees with **Stata**, whose ``mlogit`` defaults
           to the **most frequent** category as the base.  If you are migrating a
           Stata ``mlogit`` specification, pin ``base=`` explicitly (or set
           Stata's ``baseoutcome()``) so the two agree — otherwise coefficients
           and marginal effects will differ purely because of the baseline
           convention, not the estimation.
    cov_type : str, default "HC2"
        Covariance estimator type.  One of ``"nonrobust"``, ``"HC0"``–``"HC3"``.
        Ignored when *cluster* is supplied (see below).
    cluster : str, default None
        Name of a column in *data* identifying clusters.  When given,
        ``mlogit`` fits with ``cov_type="cluster"`` and passes the cluster labels
        to statsmodels via ``cov_kwds={"groups": <array>}``.  (A bare
        ``cov_type="cluster"`` string raises ``ValueError`` in statsmodels'
        ``MNLogit.fit`` — the cluster groups must be supplied this way.)

    Returns
    -------
    MultinomialResult
        Immutable result with ``(category, variable)`` coefficient / SE DataFrames,
        ``.tidy()``, ``.summary()``, ``.vcov()``, ``.predict()`` (an ``(n, K)``
        probability DataFrame), and ``.margins()`` (a ``dict`` of per-outcome
        average marginal effects).

    Examples
    --------
    >>> import open_econs as oe
    >>> r = oe.mlogit("y ~ x1 + x2", data=df, base=1)
    >>> r.tidy()
    >>> r.margins()[2]          # AMEs for outcome 2
    >>> r.predict()             # (n, K) probability DataFrame
    """
    call = _capture_call(
        formula=formula, base=base, cov_type=cov_type,
        cluster=cluster, model_type="mlogit",
    )

    cov_type = validate_cov_type(
        cov_type,
        accepted={"nonrobust", "HC0", "HC1", "HC2", "HC3"},
        estimator="mlogit()",
    )

    rhs_formula = formula.split("~", 1)[1].strip()
    lhs_name = formula.split("~", 1)[0].strip()

    from formulaic import Formula
    try:
        formula_obj = Formula(formula)
        model_spec = formula_obj.get_model_matrix(data, na_action="drop")
    except Exception as e:
        msg = str(e)
        if "not present in the dataset" in msg or "is not present" in msg:
            import re as _re
            m = _re.search(r"`(\w+)`", msg)
            bad_col = m.group(1) if m else rhs_formula
            raise errors.missing_column_error(bad_col, data.columns.tolist()) from e
        raise
    if hasattr(model_spec, "rhs"):
        XX = model_spec.rhs
    else:
        from open_econs._internal.formula import parse_formula as _parse
        _, XX = _parse(formula, data)

    # Outcome as a single label vector (NOT formulaic's dummy-matrix LHS).
    y_full = pd.Series(data[lhs_name])
    y_aligned = y_full.loc[XX.index]
    valid = y_aligned.notna().to_numpy()
    XX = XX.loc[valid]
    y_series = y_aligned.loc[valid].reset_index(drop=True)
    XX = XX.reset_index(drop=True)

    original_n = len(data)
    dropped = original_n - len(y_series)
    if dropped > 0:
        import warnings as _w
        _w.warn(
            errors.rows_dropped_warning(dropped, original_n, []),
            RuntimeWarning,
            stacklevel=3,
        )

    if len(y_series) == 0:
        raise errors.empty_data_error(original_n, dropped, [])

    cats = sorted(y_series.dropna().unique())
    if len(cats) < 2:
        raise ValueError(
            f"mlogit() requires at least 2 distinct outcome categories in '{lhs_name}'. "
            f"Found {len(cats)}: {cats}."
        )
    if base is not None and base not in cats:
        raise ValueError(
            f"base={base!r} is not one of the outcome categories {cats} in '{lhs_name}'."
        )
    base_cat = base if base is not None else cats[0]
    # Put the base first so statsmodels' integer coding uses 0 for the base.
    ordered = [base_cat] + [c for c in cats if c != base_cat]
    code_map = {c: i for i, c in enumerate(ordered)}
    y_codes = y_series.map(code_map).to_numpy(dtype=int)

    # Cluster groups must align with the surviving rows.
    cluster_arr = None
    if cluster is not None:
        if cluster not in data.columns:
            raise errors.cluster_column_error(cluster, data.columns.tolist())
        cluster_arr = data[cluster].loc[XX.index].loc[valid].to_numpy()

    _check_collinearity(XX)

    cov_type_eff, cov_kwds = _build_mnlogit_cov(cov_type, cluster, cluster_arr)

    fit_kwargs = {"disp": False, "cov_type": cov_type_eff}
    if cov_kwds:
        fit_kwargs["cov_kwds"] = cov_kwds
    fitted = sm.MNLogit(y_codes, XX.values).fit(**fit_kwargs, **kwargs)

    non_base = ordered[1:]
    var_names = list(XX.columns)
    zcrit = _norm.ppf(0.975)

    coef = np.asarray(fitted.params, dtype=float)        # (p, K-1) variable-major
    se = np.asarray(fitted.bse, dtype=float)             # (p, K-1)
    z = np.asarray(fitted.tvalues, dtype=float)
    p = np.asarray(fitted.pvalues, dtype=float)

    # Store as (category, variable) DataFrames for human readability.
    coef_df = pd.DataFrame(coef.T, index=non_base, columns=var_names)
    se_df = pd.DataFrame(se.T, index=non_base, columns=var_names)
    z_df = pd.DataFrame(z.T, index=non_base, columns=var_names)
    p_df = pd.DataFrame(p.T, index=non_base, columns=var_names)
    lower = coef_df.values - zcrit * se_df.values
    upper = coef_df.values + zcrit * se_df.values
    ci_df = pd.DataFrame(
        {
            "lower": lower.flatten(),
            "upper": upper.flatten(),
        },
        index=pd.MultiIndex.from_product([non_base, var_names]),
    )

    proba = fitted.predict(XX.values)
    fitted_df = pd.DataFrame(
        proba,
        index=XX.index,
        columns=[str(c) for c in ordered],
    )

    return MultinomialResult(
        formula=formula,
        rhs_formula=rhs_formula,
        nobs=int(fitted.nobs),
        df_resid=int(fitted.df_resid),
        df_model=int(fitted.df_model),
        cov_type=cov_type_eff,
        categories=ordered,
        base_category=base_cat,
        non_base_categories=non_base,
        variable_names=var_names,
        coefficients=coef_df,
        std_errors=se_df,
        z_stats=z_df,
        p_values=p_df,
        conf_int=ci_df,
        llf=float(fitted.llf),
        aic=float(fitted.aic),
        bic=float(fitted.bic),
        pseudo_r2=float(fitted.prsquared),
        fitted=fitted_df,
        call=call,
        _fit=fitted,
    )


def _build_mnlogit_cov(
    cov_type: str,
    cluster: str | None,
    cluster_arr: np.ndarray | None,
) -> tuple[str, dict]:
    """Translate (cov_type, cluster) into statsmodels MNLogit.fit kwargs.

    statsmodels' ``MNLogit.fit`` raises ``ValueError`` for a bare
    ``cov_type="cluster"`` — the cluster groups must be passed through
    ``cov_kwds={"groups": <array>}``.  When *cluster* is supplied we force
    ``cov_type="cluster"`` and build that dict; otherwise *cov_type* is passed
    through as-is (``nonrobust`` / ``HC0``–``HC3``).
    """
    if cluster is not None:
        return "cluster", {"groups": np.asarray(cluster_arr)}
    return cov_type, {}


def _check_collinearity(XX: pd.DataFrame) -> None:
    from numpy.linalg import matrix_rank
    X_vals = XX.values
    if matrix_rank(X_vals) < X_vals.shape[1]:
        raise errors.singular_matrix_error()
