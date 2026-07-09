from __future__ import annotations

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from open_econs._version import __version__
from open_econs._internal import errors
from open_econs.core.base import BaseModel


class DiDResult(BaseModel):
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
        _sm_fit: Any = None,
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
        self._sm_fit = _sm_fit

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
        if self._sm_fit is None:
            raise RuntimeError("vcov() requires a fitted statsmodels result.")
        return pd.DataFrame(
            self._sm_fit.cov_params(),
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
) -> DiDResult:
    call = _capture_call(
        formula=formula,
        treatment=treatment,
        post=post,
        cluster=cluster,
        cov_type=cov_type,
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

    kwargs: dict[str, Any] = {}
    if cluster is not None:
        if cluster not in data.columns:
            raise errors.missing_column_error(cluster, data.columns.tolist())
        kwargs["cov_type"] = "cluster"
        kwargs["cov_kwds"] = {"groups": data.loc[XX.index, cluster].values}
    else:
        cov_map = {
            "nonrobust": "nonrobust",
            "HC0": "HC0",
            "HC1": "HC1",
            "HC2": "HC2",
            "HC3": "HC3",
            "robust": "HC2",
        }
        kwargs["cov_type"] = cov_map.get(cov_type, "HC2")

    fitted = sm.OLS(y_arr, X_arr).fit(**kwargs)

    did_idx = list(XX.columns).index(did_term)
    did_coef = float(fitted.params[did_idx])
    did_se = float(fitted.bse[did_idx])
    did_t = float(fitted.tvalues[did_idx])
    did_p = float(fitted.pvalues[did_idx])

    coef_arr = fitted.params
    se_arr = fitted.bse
    t_arr = fitted.tvalues
    p_arr = fitted.pvalues
    conf_arr = fitted.conf_int()

    conf_int = pd.DataFrame(
        {"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]},
        index=XX.columns,
    )

    display_cov = cluster if cluster else cov_type

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
        _sm_fit=fitted,
    )


class EventStudyResult(BaseModel):
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
        _sm_fit: Any = None,
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
        self._sm_fit = _sm_fit

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
        if self._sm_fit is None:
            raise RuntimeError("vcov() requires a fitted statsmodels result.")
        return pd.DataFrame(
            self._sm_fit.cov_params(),
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
) -> EventStudyResult:
    call = _capture_call(
        formula=formula,
        treatment=treatment,
        post=post,
        cluster=cluster,
        cov_type=cov_type,
        omitted_period=omitted_period,
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

    event_fml = f"{dep_var} ~ C({treatment}_event_cat, Treatment('{omitted_period}'))"
    cov_rhs = rhs.replace(treatment, "")
    cov_rhs = cov_rhs.replace(f"+ {post}", "").replace(f"{post} +", "").replace(post, "").strip()
    cov_rhs = cov_rhs.strip(" +")
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

    kwargs: dict[str, Any] = {}
    if cluster is not None:
        if cluster not in data.columns:
            raise errors.missing_column_error(cluster, data.columns.tolist())
        kwargs["cov_type"] = "cluster"
        kwargs["cov_kwds"] = {"groups": data.loc[XX.index, cluster].values}
    else:
        cov_map = {
            "nonrobust": "nonrobust",
            "HC0": "HC0",
            "HC1": "HC1",
            "HC2": "HC2",
            "HC3": "HC3",
            "robust": "HC2",
        }
        kwargs["cov_type"] = cov_map.get(cov_type, "HC2")

    fitted = sm.OLS(y_arr, X_arr).fit(**kwargs)

    coef_arr = fitted.params.values
    se_arr = fitted.bse.values
    t_arr = fitted.tvalues.values
    p_arr = fitted.pvalues.values
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
        raw_name = c.replace(f"C({treatment}_event_cat, Treatment('{omitted_period}'))[T.", "").rstrip("]")
        try:
            period = float(raw_name)
        except ValueError:
            period = float("nan")
        event_periods.append(period)
        idx = list(XX.columns).index(c)
        event_coefs.append(float(fitted.params[idx]))
        lower, upper = conf_arr[idx]
        event_lower.append(float(lower))
        event_upper.append(float(upper))

    ev_df = pd.DataFrame({
        "period": event_periods,
        "coef": event_coefs,
        "ci_lower": event_lower,
        "ci_upper": event_upper,
    }).sort_values("period").reset_index(drop=True)

    display_cov = cluster if cluster else cov_type

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
        _sm_fit=fitted,
    )


def _capture_call(**kwargs: Any) -> dict[str, Any]:
    kwargs["timestamp"] = str(datetime.now())
    kwargs["package_version"] = __version__
    return kwargs