"""Ordered logit / ordered probit estimators (``ologit`` / ``oprobit``).

Backend: ``statsmodels.miscmodels.ordinal_model.OrderedModel`` (same
statsmodels pattern as ``open_econs/models/discrete/logit.py``). The MLE is
polished with a high-precision L-BFGS-B pass so the point estimates, cutpoints,
and OIM standard errors match Stata ``ologit`` / ``oprobit`` and R
``MASS::polr`` to ``1e-6`` (the raw statsmodels optimizer stops ~3e-5 short of
Stata; see ``methodology/limited/ordered.md`` root cause).

Supports ``formula`` + ``data``, ``cov_type`` in
``{nonrobust, HC0, HC1, HC2, HC3}``, and ``distr`` (``"logit"`` / ``"probit"``).
No fixed-effects arguments: Stata/R base ``ologit`` / ``oprobit`` have none.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from open_econs.core.call_capture import capture_call as _capture_call
from open_econs._internal import errors
from open_econs.core.results import OrderedResult
from open_econs.core.cov_type import validate_cov_type


def _fit_ordered(
    formula: str,
    data: pd.DataFrame,
    distr: str,
    cov_type: str,
) -> OrderedResult:
    """Shared fitting routine for ``ologit`` / ``oprobit``."""
    call = _capture_call(formula=formula, cov_type=cov_type, model_type=distr)

    cov_type = validate_cov_type(
        cov_type,
        accepted={"nonrobust", "HC0", "HC1", "HC2", "HC3"},
        estimator=f"{distr}()",
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

    # OrderedModel supplies its own thresholds; drop any intercept column so
    # statsmodels does not raise "There should not be a constant in the model".
    intercept_cols = [c for c in XX.columns if str(c).strip() in ("Intercept", "const", "1")]
    if intercept_cols:
        XX = XX.drop(columns=intercept_cols)

    original_n = len(data)
    dropped = original_n - len(yy)
    if dropped > 0:
        warnings.warn(
            errors.rows_dropped_warning(dropped, original_n, []),
            RuntimeWarning,
            stacklevel=3,
        )
    if len(yy) == 0:
        raise errors.empty_data_error(original_n, dropped, [])

    y_series = yy.iloc[:, 0] if hasattr(yy, "iloc") else pd.Series(yy)
    y_clean = pd.Series(y_series).reset_index(drop=True)
    X_clean = XX.reset_index(drop=True)

    cats = sorted(y_clean.dropna().astype(int).unique().tolist())
    if len(cats) < 3:
        raise ValueError(
            f"{distr}() requires an ordered dependent variable with at least 3 "
            f"distinct integer levels; found {len(cats)}: {cats}."
        )
    y_cat = pd.Series(
        pd.Categorical(y_clean.astype(int).astype(str), categories=[str(c) for c in cats], ordered=True)
    )

    endog_name = y_clean.name if y_clean.name is not None else "y"

    from statsmodels.miscmodels.ordinal_model import OrderedModel
    model = OrderedModel(y_cat, X_clean, distr=distr)

    # 1) statsmodels default start (Nelder-Mead) for a stable starting point.
    start_fit = model.fit(disp=False, method="nm", maxiter=1000)
    start_params = np.asarray(start_fit.params, dtype=float)

    # 2) High-precision polish so point estimates / cutpoints / OIM SEs match
    #    Stata & R to 1e-6. The raw statsmodels optimizer converges ~3e-5 short.
    def _nll(p: np.ndarray) -> float:
        return float(-model.loglike(p))

    polished = minimize(
        _nll, start_params, method="L-BFGS-B",
        options={"gtol": 1e-12, "ftol": 1e-14, "maxiter": 10000},
    )
    params = polished.x

    # Cutpoints in Stata convention (cumulative thresholds, increasing).
    thr = model.transform_threshold_params(params)[1:-1]
    coef_names = list(X_clean.columns)
    cut_names = [f"cut{j + 1}" for j in range(len(thr))]
    all_names = coef_names + cut_names

    # Covariance: OIM from -hessian, or HC sandwich from numerical scores.
    bread = np.linalg.inv(-model.hessian(params))
    if cov_type == "nonrobust":
        cov = bread
    else:
        cov = _sandwich_cov(model, params, bread, cov_type)

    coef_arr = params[: len(coef_names)]
    se_arr = np.sqrt(np.diag(cov))
    n = int(model.nobs)
    k_total = len(params)

    z_arr = coef_arr / se_arr[: len(coef_names)]
    p_arr = 2.0 * _stats_norm_sf(np.abs(z_arr))
    crit = _stats_norm_ppf(0.975)
    conf_arr = np.column_stack([
        coef_arr - crit * se_arr[: len(coef_names)],
        coef_arr + crit * se_arr[: len(coef_names)],
    ])

    coefficients = pd.Series(coef_arr, index=coef_names)
    cutpoints = pd.Series(thr, index=cut_names)

    conf_int = pd.DataFrame(
        {"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]},
        index=coef_names,
    )
    std_errors = pd.Series(se_arr[: len(coef_names)], index=coef_names)

    llf = float(-_nll(params))
    df_model = len(coef_names)
    df_resid = n - k_total

    # Fitted class probabilities (in-sample).
    proba = model.predict(params, which="prob")
    fitted_probs = pd.DataFrame(
        proba, index=X_clean.index,
        columns=[str(c) for c in cats],
    )
    fitted_class = pd.Series(
        np.array(cats)[np.argmax(proba, axis=1)],
        index=X_clean.index, name="predicted_class",
    )

    return OrderedResult(
        formula=formula,
        rhs_formula=rhs_formula,
        nobs=n,
        df_resid=df_resid,
        df_model=df_model,
        cov_type=cov_type,
        distr=distr,
        endog_name=endog_name,
        categories=cats,
        coefficients=coefficients,
        cutpoints=cutpoints,
        std_errors=std_errors,
        z_stats=pd.Series(z_arr, index=coef_names),
        p_values=pd.Series(p_arr, index=coef_names),
        conf_int=conf_int,
        llf=llf,
        fitted_probs=fitted_probs,
        fitted_class=fitted_class,
        call=call,
        model_type=distr,
        _cov=pd.DataFrame(cov, index=all_names, columns=all_names),
        _fit=model,
        _params=params,
    )


def _sandwich_cov(model: Any, params: np.ndarray, bread: np.ndarray, cov_type: str) -> np.ndarray:
    """HC0/HC1/HC2/HC3 robust covariance from numerical observation scores."""
    from statsmodels.tools.numdiff import approx_fprime
    scores = approx_fprime(params, model.loglikeobs, centered=True, epsilon=1e-8)
    scores = np.asarray(scores, dtype=float)
    n, k = scores.shape
    xtx_inv = np.linalg.inv(scores.T @ scores)
    if cov_type == "HC0":
        w = np.ones(n)
    elif cov_type == "HC1":
        w = np.full(n, n / (n - k))
    elif cov_type in ("HC2", "HC3"):
        h = np.einsum("ni,ij,jn->n", scores, xtx_inv, scores.T)
        if cov_type == "HC2":
            w = 1.0 / (1.0 - h)
        else:
            w = 1.0 / (1.0 - h) ** 2
    else:  # pragma: no cover - guarded by validate_cov_type
        raise ValueError(f"Unsupported cov_type for sandwich: {cov_type}")
    meat = (scores * w[:, None]).T @ scores
    return bread @ meat @ bread


def _stats_norm_sf(x: np.ndarray) -> np.ndarray:
    from scipy import stats as _st
    return _st.norm.sf(x)


def _stats_norm_ppf(q: float) -> float:
    from scipy import stats as _st
    return float(_st.norm.ppf(q))


def ologit(formula: str, data: pd.DataFrame, cov_type: str = "nonrobust") -> OrderedResult:
    """Estimate an ordered logit (proportional-odds) model.

    Wraps ``statsmodels.miscmodels.ordinal_model.OrderedModel`` with the
    ``logit`` link, polished to match Stata ``ologit`` and R ``MASS::polr``
    (``method = "logistic"``) to a numeric tolerance of ``1e-6`` on point
    estimates, cutpoints, and OIM standard errors.

    Parameters
    ----------
    formula : str
        Two-sided formula, e.g. ``"y ~ x1 + x2 + x3"``. The left-hand side must
        be an integer-coded ordered variable with at least 3 levels.
    data : pd.DataFrame
        Data containing all formula variables.
    cov_type : str, default "nonrobust"
        Covariance estimator. One of ``"nonrobust"`` (OIM), ``"HC0"``,
        ``"HC1"``, ``"HC2"``, ``"HC3"``.

    Returns
    -------
    OrderedResult
        Immutable result with ``.coefficients``, ``.cutpoints``, ``.tidy()``,
        ``.summary()``, ``.predict()`` (class probabilities), and ``.margins()``.

    Examples
    --------
    >>> import open_econs as oe
    >>> r = oe.ologit("y ~ x1 + x2 + x3", data=df)
    >>> r.tidy(); r.cutpoints; r.predict(type="probs")
    """
    return _fit_ordered(formula, data, distr="logit", cov_type=cov_type)


def oprobit(formula: str, data: pd.DataFrame, cov_type: str = "nonrobust") -> OrderedResult:
    """Estimate an ordered probit model.

    Wraps ``statsmodels.miscmodels.ordinal_model.OrderedModel`` with the
    ``probit`` link, polished to match Stata ``oprobit`` and R ``MASS::polr``
    (``method = "probit"``) to a numeric tolerance of ``1e-6`` on point
    estimates, cutpoints, and OIM standard errors.

    Parameters
    ----------
    formula : str
        Two-sided formula, e.g. ``"y ~ x1 + x2 + x3"``.
    data : pd.DataFrame
        Data containing all formula variables.
    cov_type : str, default "nonrobust"
        Covariance estimator. One of ``"nonrobust"``, ``"HC0"``, ``"HC1"``,
        ``"HC2"``, ``"HC3"``.

    Returns
    -------
    OrderedResult
        Immutable result with ``.coefficients``, ``.cutpoints``, ``.tidy()``,
        ``.summary()``, ``.predict()``, and ``.margins()``.
    """
    return _fit_ordered(formula, data, distr="probit", cov_type=cov_type)
