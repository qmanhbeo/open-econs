from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from open_econs._version import __version__
from open_econs.core.call_capture import capture_call as _capture_call
from open_econs._internal import errors
from open_econs.core.base import BaseModel
from open_econs.core.cov_type import validate_cov_type


class DiDResult(BaseModel):
    """Result of a two-period difference-in-differences estimator.

    Immutable result exposing a uniform interface: ``.tidy()`` (coefficients,
    SEs, t-stats, p-values, CI), ``.summary()`` (text), ``.export()``
    (CSV/JSON/Pickle), ``.vcov()``, ``.to_latex()`` / ``.to_html()``.  The
    key DiD quantity is the treatment-on-treated effect ``did_coef`` with its
    ``did_std_err`` / ``did_t_stat`` / ``did_p_value``.
    """
    def __init__(
        self,
        *,
        formula: str,
        nobs: int,
        dep_var: str,
        treatment_var: str,
        post_var: str,
        cluster_var: str | None,
        cov_type: str,
        coefficients: pd.Series,
        std_errors: pd.Series,
        t_stats: pd.Series,
        p_values: pd.Series,
        conf_int: pd.DataFrame,
        did_coef: float,
        did_std_err: float,
        did_t_stat: float,
        did_p_value: float,
        r_squared: float,
        adj_r_squared: float,
        rsd: float,
        call: dict[str, Any],
        _fit: Any = None,
    ) -> None:
        self.formula = formula
        self.data_shape = (nobs, coefficients.shape[0])
        self.cov_type = cov_type
        self.call = call
        self.timestamp = datetime.now()
        self.package_version = __version__

        self.nobs = nobs
        self.dep_var = dep_var
        self.treatment_var = treatment_var
        self.post_var = post_var
        self.cluster_var = cluster_var
        self.coefficients = coefficients
        self.std_errors = std_errors
        self.t_stats = t_stats
        self.p_values = p_values
        self.conf_int = conf_int
        self.did_coefficient = did_coef
        self.did_std_error = did_std_err
        self.did_t_stat = did_t_stat
        self.did_p_value = did_p_value
        self.r_squared = r_squared
        self.adj_r_squared = adj_r_squared
        self.rsd = rsd
        self._fit = _fit

        self._freeze()

    def tidy(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "Variable": self.coefficients.index,
            "Coef": self.coefficients.values,
            "Std Err": self.std_errors.values,
            "t": self.t_stats.values,
            "P>|t|": self.p_values.values,
            "0.025": self.conf_int["lower"].values,
            "0.975": self.conf_int["upper"].values,
        })
        df.index.name = None
        return df

    def summary(self) -> str:
        di_str = (
            f"Difference-in-Differences Estimate: {self.did_coefficient:.6f}\n"
            f"Std. Error:                          {self.did_std_error:.6f}\n"
            f"t-statistic:                         {self.did_t_stat:.4f}\n"
            f"P>|t|:                               {self.did_p_value:.6e}\n"
        )
        header = (
            f"                  Difference-in-Differences Results                    \n"
            f"======================================================================\n"
            f"Dep. Variable:               {self.dep_var}\n"
            f"Treatment variable:          {self.treatment_var}\n"
            f"Post variable:               {self.post_var}\n"
            f"Cluster:                     {self.cluster_var if self.cluster_var else 'none'}\n"
            f"No. Observations:            {self.nobs}\n"
            f"R-squared:                   {self.r_squared:.6f}\n"
            f"Adj. R-squared:              {self.adj_r_squared:.6f}\n"
            f"Covariance Type:             {self.cov_type}\n"
            f"======================================================================\n"
            f"DiD coefficient ({self.treatment_var}#{self.post_var}):\n"
            f"{di_str}"
            f"======================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return (
            header + tbl +
            "\n======================================================================\n"
        )

    def vcov(self) -> pd.DataFrame:
        if self._fit is None:
            raise RuntimeError("vcov() requires a fitted statsmodels result.")
        return pd.DataFrame(
            self._fit.cov_params(),
            index=self.coefficients.index,
            columns=self.coefficients.index,
        )


def did(
    formula: str,
    data: pd.DataFrame,
    treatment: str,
    post: str,
    cluster: str | None = None,
    cov_type: str = "HC2",
    lags: int | None = None,
    time: str | None = None,
    hac_adjust: bool = False,
) -> DiDResult:
    """Two-period difference-in-differences (interactive fixed-effects) estimator.

    Estimates the canonical DiD specification with a treatment x post
    interaction term.  The coefficient on ``treatment:post`` is the average
    treatment effect on the treated (ATT).  Supports heteroskedasticity-robust,
    cluster-robust, or Newey-West HAC standard errors and arbitrary additional
    controls.

    Parameters
    ----------
    formula : str
        Two-sided formula, e.g. ``"y ~ treatment * post + x1"``.  The
        interaction ``treatment:post`` is read off automatically.
    data : pd.DataFrame
        Analysis data.
    treatment : str
        Name of the binary treatment indicator (1 = treated).
    post : str
        Name of the binary post-treatment indicator (1 = post period).
    cluster : str, optional
        Column to cluster standard errors by (e.g. unit or group id).
    cov_type : str, default "HC2"
        Covariance type: ``"nonrobust"``, ``"HC0"``, ``"HC1"``, ``"HC2"``,
        ``"HC3"``, or ``"HAC"``.  ``"robust"`` aliases to ``"HC2"`` and
        ``"hac"`` (any case) aliases to ``"HAC"``.  Ignored when ``cluster``
        is given.
    lags : int, optional
        Number of Newey-West lags (required when ``cov_type="HAC"``).
    time : str, optional
        Column name for the time-period index used by HAC period-aggregation.
        Required when ``cov_type="HAC"``.
    hac_adjust : bool, default False
        Apply the ``N / (N - K)`` finite-sample correction to HAC variances.

    Returns
    -------
    DiDResult
        Immutable result with ``.tidy()``, ``.summary()``, ``.export()``.
    """
    call = _capture_call(
        formula=formula,
        treatment=treatment,
        post=post,
        cluster=cluster,
        cov_type=cov_type,
        lags=lags,
        time=time,
        hac_adjust=hac_adjust,
    )

    cov_type = validate_cov_type(
        cov_type,
        accepted={"nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"},
        aliases={"robust": "HC2"},
        estimator="did()",
    )

    from formulaic import Formula

    try:
        formula_obj = Formula(formula)
        model_spec = formula_obj.get_model_matrix(data, na_action="drop")
    except Exception as e:
        msg = str(e)
        if "not present in the dataset" in msg or "is not present" in msg:
            import re as _re
            m = _re.search(r"`(\w+)`", msg)
            bad_col = m.group(1) if m else formula
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
    if dropped > 0:
        import warnings as _w
        _w.warn(
            errors.rows_dropped_warning(dropped, original_n, []),
            RuntimeWarning, stacklevel=3,
        )

    if len(yy) == 0:
        raise errors.empty_data_error(original_n, dropped, [])

    y_arr = yy.values.ravel().astype(float)
    X_arr = XX.values.astype(float)

    did_term = f"{treatment}:{post}"
    if did_term not in XX.columns and f"{treatment}:{post}" not in XX.columns:
        alt = f"{post}:{treatment}"
        if alt in XX.columns:
            did_term = alt
        elif f"{post}:{treatment}" in XX.columns:
            did_term = f"{post}:{treatment}"
        else:
            raise ValueError(
                f"Interaction term '{treatment}:{post}' not found in design matrix. "
                f"Available columns: {list(XX.columns)}"
            )

    import statsmodels.api as sm

    use_hac = cov_type == "HAC"

    if use_hac:
        if lags is None:
            raise ValueError(
                "Newey-West HAC requires `lags` (e.g. lags=1)."
            )
        if time is None:
            raise ValueError(
                "Newey-West HAC requires `time` (the time dimension used "
                "as the Newey-West period index)."
            )
        from open_econs.core.cov import newey_west_cov, _as_int_labels
        fitted = sm.OLS(y_arr, X_arr).fit(cov_type="nonrobust")
        time_labels = _as_int_labels(data.loc[XX.index, time].values)
        V_cov = newey_west_cov(
            X_arr, np.asarray(fitted.resid), max_lags=lags,
            cluster=time_labels, adjust=hac_adjust,
        )
        se_arr = np.sqrt(np.maximum(np.diag(V_cov), 0.0))
        display_cov = f"HAC({lags})"
    elif cluster is not None:
        if cluster not in data.columns:
            raise errors.missing_column_error(cluster, data.columns.tolist())
        fitted = sm.OLS(y_arr, X_arr).fit(
            cov_type="cluster",
            cov_kwds={"groups": data.loc[XX.index, cluster].values},
        )
        se_arr = fitted.bse
        display_cov = cluster
    else:
        fitted = sm.OLS(y_arr, X_arr).fit(cov_type=cov_type)
        se_arr = fitted.bse
        display_cov = cov_type

    coef_arr = fitted.params

    if use_hac:
        t_arr = np.where(se_arr > 0, coef_arr / se_arr, np.nan)
        from scipy.stats import norm as _norm
        p_arr = 2.0 * (1.0 - _norm.cdf(np.abs(t_arr)))
        conf_arr = np.column_stack(
            [coef_arr - 1.96 * se_arr, coef_arr + 1.96 * se_arr]
        )
    else:
        t_arr = fitted.tvalues
        p_arr = fitted.pvalues
        conf_arr = fitted.conf_int()

    did_idx = list(XX.columns).index(did_term)
    did_coef = float(coef_arr[did_idx])
    did_se = float(se_arr[did_idx])
    did_t = float(t_arr[did_idx])
    did_p = float(p_arr[did_idx])

    conf_int = pd.DataFrame(
        {"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]},
        index=XX.columns,
    )

    return DiDResult(
        formula=formula,
        nobs=int(fitted.nobs),
        dep_var=formula.split("~")[0].strip(),
        treatment_var=treatment,
        post_var=post,
        cluster_var=cluster,
        cov_type=display_cov,
        coefficients=pd.Series(coef_arr, index=XX.columns),
        std_errors=pd.Series(se_arr, index=XX.columns),
        t_stats=pd.Series(t_arr, index=XX.columns),
        p_values=pd.Series(p_arr, index=XX.columns),
        conf_int=conf_int,
        did_coef=did_coef,
        did_std_err=did_se,
        did_t_stat=did_t,
        did_p_value=did_p,
        r_squared=float(fitted.rsquared),
        adj_r_squared=float(fitted.rsquared_adj),
        rsd=float(np.sqrt(fitted.scale)),
        call=call,
        _fit=fitted,
    )


class EventStudyResult(BaseModel):
    """Result of a relative-time event-study (dynamic DiD) estimator.

    Immutable result with the uniform interface (``.tidy()``, ``.summary()``,
    ``.export()``, ``.to_latex()`` / ``.to_html()``).  ``event_times`` lists
    the relative periods, ``coefficients`` / ``std_errors`` / ``p_values`` hold
    the per-period estimates, and ``plot()`` draws the event-study path when
    matplotlib is installed (otherwise returns ``None``).
    """
    def __init__(
        self,
        *,
        formula: str,
        nobs: int,
        dep_var: str,
        event_time_col: str,
        cov_type: str,
        coefficients: pd.Series,
        std_errors: pd.Series,
        t_stats: pd.Series,
        p_values: pd.Series,
        conf_int: pd.DataFrame,
        event_coefficients: pd.DataFrame,
        r_squared: float,
        adj_r_squared: float,
        rsd: float,
        call: dict[str, Any],
        _fit: Any = None,
    ) -> None:
        self.formula = formula
        self.data_shape = (nobs, coefficients.shape[0])
        self.cov_type = cov_type
        self.call = call
        self.timestamp = datetime.now()
        self.package_version = __version__

        self.nobs = nobs
        self.dep_var = dep_var
        self.event_var_col = event_time_col
        self.coefficients = coefficients
        self.std_errors = std_errors
        self.t_stats = t_stats
        self.p_values = p_values
        self.conf_int = conf_int
        self.event_coefficients = event_coefficients
        self.r_squared = r_squared
        self.adj_r_squared = adj_r_squared
        self.rsd = rsd
        self._fit = _fit

        self._freeze()

    def tidy(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "Variable": self.coefficients.index,
            "Coef": self.coefficients.values,
            "Std Err": self.std_errors.values,
            "t": self.t_stats.values,
            "P>|t|": self.p_values.values,
            "0.025": self.conf_int["lower"].values,
            "0.975": self.conf_int["upper"].values,
        })
        df.index.name = None
        return df

    def summary(self) -> str:
        header = (
            f"                  Event-Study Regression Results                        \n"
            f"======================================================================\n"
            f"Dep. Variable:               {self.dep_var}\n"
            f"No. Observations:            {self.nobs}\n"
            f"R-squared:                   {self.r_squared:.6f}\n"
            f"Adj. R-squared:              {self.adj_r_squared:.6f}\n"
            f"======================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return (
            header + tbl +
            "\n======================================================================\n"
        )

    def plot(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "plot() requires matplotlib. Install it with: "
                "pip install open-econs[plot]  or  pip install matplotlib"
            )
        ev = self.event_coefficients
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(-0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.errorbar(
            ev["period"], ev["coef"],
            yerr=ev["ci_upper"] - ev["coef"],
            fmt="o-", capsize=3, capthick=1, markersize=5,
        )
        ax.fill_between(
            ev["period"], ev["ci_lower"], ev["ci_upper"],
            alpha=0.15,
        )
        ax.set_xlabel("Event time")
        ax.set_ylabel("Coefficient")
        ax.set_title("Event-study coefficients")
        fig.tight_layout()
        plt.show()

    def vcov(self) -> pd.DataFrame:
        if self._fit is None:
            raise RuntimeError("vcov() requires a fitted statsmodels result.")
        return pd.DataFrame(
            self._fit.cov_params(),
            index=self.coefficients.index,
            columns=self.coefficients.index,
        )


def event_study(
    formula: str,
    data: pd.DataFrame,
    treatment: str,
    post: str,
    cluster: str | None = None,
    cov_type: str = "HC2",
    omitted_period: int = -1,
    lags: int | None = None,
    time: str | None = None,
    hac_adjust: bool = False,
) -> EventStudyResult:
    """Event-study (relative-time) difference-in-differences estimator.

    Estimates a dynamic DiD with one dummy per relative time-to-treatment
    period, revealing the pre-trend and the evolution of the treatment effect.
    The data must contain a column ``"{treatment}_event_time"`` whose values
    are periods relative to treatment (e.g. -2, -1, 0, 1, 2), with
    ``omitted_period`` (default -1, the period just before treatment) used as
    the normalization baseline.  Supports Newey-West HAC standard errors.

    Parameters
    ----------
    formula : str
        Two-sided formula, e.g. ``"y ~ x1"``.  The relative-time dummies are
        generated automatically from the ``"{treatment}_event_time"`` column.
    data : pd.DataFrame
        Analysis data, including the ``"{treatment}_event_time"`` column.
    treatment : str
        Name of the treatment indicator (used to build the event-time column).
    post : str
        Name of the post-treatment indicator (used only for diagnostics).
    cluster : str, optional
        Column to cluster standard errors by.
    cov_type : str, default "HC2"
        Covariance type: ``"nonrobust"``, ``"HC0"``, ``"HC1"``, ``"HC2"``,
        ``"HC3"``, or ``"HAC"``.  ``"robust"`` aliases to ``"HC2"`` and
        ``"hac"`` (any case) aliases to ``"HAC"``.  Ignored when ``cluster``
        is given.
    omitted_period : int, default -1
        Relative period held out as the baseline in the event-study graph.
    lags : int, optional
        Number of Newey-West lags (required when ``cov_type="HAC"``).
    time : str, optional
        Column name for the time-period index used by HAC period-aggregation.
        Required when ``cov_type="HAC"``.
    hac_adjust : bool, default False
        Apply the ``N / (N - K)`` finite-sample correction to HAC variances.

    Returns
    -------
    EventStudyResult
        Immutable result; ``.tidy()`` shows each relative-period coefficient,
        ``.plot()`` renders the event-study path if matplotlib is available.
    """
    call = _capture_call(
        formula=formula,
        treatment=treatment,
        post=post,
        cluster=cluster,
        cov_type=cov_type,
        omitted_period=omitted_period,
        lags=lags,
        time=time,
        hac_adjust=hac_adjust,
    )

    cov_type = validate_cov_type(
        cov_type,
        accepted={"nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"},
        aliases={"robust": "HC2"},
        estimator="event_study()",
    )

    from formulaic import Formula

    dep_var = formula.split("~")[0].strip()
    rhs = formula.split("~", 1)[1].strip()

    if treatment not in data.columns:
        raise errors.missing_column_error(treatment, data.columns.tolist())

    event_col = f"{treatment}_event_time"
    if event_col not in data.columns:
        unique_times = sorted(data[post].unique())
        periods = [t - omitted_period for t in unique_times]
        raise ValueError(
            f"Column '{event_col}' not found in data. "
            f"For event-study, create a column '{event_col}' indicating "
            f"periods relative to treatment (e.g., -2, -1, 0, 1, 2...). "
            f"Your unique `{post}` values: {unique_times}. "
            f"Suggested event-time mapping: {periods}"
        )

    data = data.copy()
    data[f"{treatment}_event_cat"] = data[event_col].astype("category")

    unique_periods = sorted(data[event_col].dropna().unique())
    if omitted_period not in unique_periods:
        omitted_period = unique_periods[0]

    cov_rhs = _covariates_excluding(rhs, treatment, post)
    event_fml = f"{dep_var} ~ C({treatment}_event_cat, Treatment({omitted_period}))"
    if cov_rhs:
        event_fml = f"{event_fml} + {cov_rhs}"

    try:
        formula_obj = Formula(event_fml)
        model_spec = formula_obj.get_model_matrix(data, na_action="drop")
    except Exception as e:
        msg = str(e)
        if "not present in the dataset" in msg or "is not present" in msg:
            import re as _re
            m = _re.search(r"`(\w+)`", msg)
            bad_col = m.group(1) if m else event_fml
            raise errors.missing_column_error(bad_col, data.columns.tolist()) from e
        raise

    if hasattr(model_spec, "rhs"):
        XX = model_spec.rhs
        yy = model_spec.lhs
    else:
        from open_econs._internal.formula import parse_formula as _parse
        yy, XX = _parse(event_fml, data)
        model_spec = None

    original_n = len(data)
    dropped = original_n - len(yy)
    if dropped > 0:
        import warnings as _w
        _w.warn(
            errors.rows_dropped_warning(dropped, original_n, []),
            RuntimeWarning, stacklevel=3,
        )

    if len(yy) == 0:
        raise errors.empty_data_error(original_n, dropped, [])

    y_arr = yy.values.ravel().astype(float)
    X_arr = XX.values.astype(float)

    event_cols = [c for c in XX.columns if c.startswith(f"C({treatment}_event_cat,")]
    if not event_cols:
        raise ValueError("No event-time indicators found in design matrix.")

    import statsmodels.api as sm

    use_hac = cov_type == "HAC"

    if use_hac:
        if lags is None:
            raise ValueError(
                "Newey-West HAC requires `lags` (e.g. lags=1)."
            )
        if time is None:
            raise ValueError(
                "Newey-West HAC requires `time` (the time dimension used "
                "as the Newey-West period index)."
            )
        from open_econs.core.cov import newey_west_cov, _as_int_labels
        fitted = sm.OLS(y_arr, X_arr).fit(cov_type="nonrobust")
        time_labels = _as_int_labels(data.loc[XX.index, time].values)
        V_cov = newey_west_cov(
            X_arr, np.asarray(fitted.resid), max_lags=lags,
            cluster=time_labels, adjust=hac_adjust,
        )
        se_arr = np.sqrt(np.maximum(np.diag(V_cov), 0.0))
        display_cov = f"HAC({lags})"
    elif cluster is not None:
        if cluster not in data.columns:
            raise errors.missing_column_error(cluster, data.columns.tolist())
        fitted = sm.OLS(y_arr, X_arr).fit(
            cov_type="cluster",
            cov_kwds={"groups": data.loc[XX.index, cluster].values, "use_t": True},
        )
        se_arr = fitted.bse
        display_cov = cluster
    else:
        fitted = sm.OLS(y_arr, X_arr).fit(
            cov_type=cov_type, cov_kwds={"use_t": True},
        )
        se_arr = fitted.bse
        display_cov = cov_type

    coef_arr = fitted.params

    if use_hac:
        t_arr = np.where(se_arr > 0, coef_arr / se_arr, np.nan)
        from scipy.stats import norm as _norm
        p_arr = 2.0 * (1.0 - _norm.cdf(np.abs(t_arr)))
        conf_arr = np.column_stack(
            [coef_arr - 1.96 * se_arr, coef_arr + 1.96 * se_arr]
        )
    else:
        t_arr = fitted.tvalues
        p_arr = fitted.pvalues
        conf_arr = fitted.conf_int()

    conf_int = pd.DataFrame(
        {"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]},
        index=XX.columns,
    )

    event_periods: list[float] = []
    event_coefs = []
    event_lower = []
    event_upper = []
    for c in event_cols:
        raw_name = c.replace(f"C({treatment}_event_cat, Treatment({omitted_period}))[T.", "").rstrip("]")
        try:
            period = float(raw_name)
        except ValueError:
            period = float("nan")
        event_periods.append(period)
        idx = list(XX.columns).index(c)
        event_coefs.append(float(coef_arr[idx]))
        lower, upper = conf_arr[idx]
        event_lower.append(float(lower))
        event_upper.append(float(upper))

    ev_df = pd.DataFrame({
        "period": event_periods,
        "coef": event_coefs,
        "ci_lower": event_lower,
        "ci_upper": event_upper,
    }).sort_values("period").reset_index(drop=True)

    return EventStudyResult(
        formula=formula,
        nobs=int(fitted.nobs),
        dep_var=dep_var,
        event_time_col=event_col,
        cov_type=display_cov,
        coefficients=pd.Series(coef_arr, index=XX.columns),
        std_errors=pd.Series(se_arr, index=XX.columns),
        t_stats=pd.Series(t_arr, index=XX.columns),
        p_values=pd.Series(p_arr, index=XX.columns),
        conf_int=conf_int,
        event_coefficients=ev_df,
        r_squared=float(fitted.rsquared),
        adj_r_squared=float(fitted.rsquared_adj),
        rsd=float(np.sqrt(fitted.scale)),
        call=call,
        _fit=fitted,
    )




def _split_top_level_plus(rhs: str) -> list[str]:
    """Split a formula RHS on `+` tokens that are not nested inside parentheses."""
    parts: list[str] = []
    depth = 0
    current = ""
    for ch in rhs:
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth > 0:
                depth -= 1
        if ch == "+" and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)
    return [p.strip() for p in parts if p.strip()]


def _covariates_excluding(rhs: str, *drop: str) -> str:
    """Return the RHS with any term referencing *drop* variables removed.

    Handles interaction terms (e.g. ``treated * post``) and covariates nested
    in functions such as ``C(...)``. Terms are matched on word boundaries so a
    covariate named ``treated_x`` is not accidentally removed by ``treated``.
    """
    import re as _re

    drop_terms = [d.strip() for d in drop if d and d.strip()]
    patterns = [
        _re.compile(r"(?<![\w])" + _re.escape(d) + r"(?![\w])")
        for d in drop_terms
    ]

    keep: list[str] = []
    for term in _split_top_level_plus(rhs):
        if any(p.search(term) for p in patterns):
            continue
        keep.append(term)
    return " + ".join(keep)