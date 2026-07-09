from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2
from scipy.stats import f as _f

from open_econs._version import __version__
from open_econs.core.base import BaseModel
from open_econs.core.results import OLSResult


class RandomEffectsResult(BaseModel):
    """Result of a panel random-effects (Glass-S Arnarsson / Swamy-Arora) estimator."""

    def __init__(
        self,
        *,
        formula: str,
        rhs_formula: str,
        nobs: int,
        n_entities: int,
        n_time: int,
        coefficients: pd.Series,
        std_errors: pd.Series,
        z_stats: pd.Series,
        p_values: pd.Series,
        conf_int: pd.DataFrame,
        r_squared: float,
        r_squared_within: float,
        r_squared_between: float,
        r_squared_overall: float,
        cov_type: str,
        theta: pd.Series,
        sigma2_effects: float,
        sigma2_eps: float,
        rho: float,
        llf: float,
        aic: float,
        bic: float,
        fitted: pd.Series,
        residuals: pd.Series,
        call: dict[str, Any],
        _lm_fit: Any = None,
    ) -> None:
        self.formula = formula
        self.rhs_formula = rhs_formula
        self.data_shape = (nobs, coefficients.shape[0])
        self.cov_type = cov_type
        self.call = call
        self.timestamp = datetime.now()
        self.package_version = __version__

        self.nobs = nobs
        self.n_entities = n_entities
        self.n_time = n_time
        self.coefficients = coefficients
        self.std_errors = std_errors
        self.z_stats = z_stats
        self.p_values = p_values
        self.conf_int = conf_int
        self.r_squared = r_squared
        self.r_squared_within = r_squared_within
        self.r_squared_between = r_squared_between
        self.r_squared_overall = r_squared_overall
        self.theta = theta
        self.sigma2_effects = sigma2_effects
        self.sigma2_eps = sigma2_eps
        self.rho = rho
        self.llf = llf
        self.aic = aic
        self.bic = bic
        self.fitted_values = fitted if fitted is not None else pd.Series(dtype=float)
        self.residuals = residuals
        self._lm_fit = _lm_fit

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
        llf_str = f"{self.llf:.3f}" if self.llf is not None and not self._isnan(self.llf) else "N/A"
        aic_str = f"{self.aic:.2f}" if self.aic is not None and not self._isnan(self.aic) else "N/A"
        bic_str = f"{self.bic:.2f}" if self.bic is not None and not self._isnan(self.bic) else "N/A"
        header = (
            f"                 Random Effects (GLS) Regression Results              \n"
            f"======================================================================\n"
            f"Dep. Variable:               {self.formula.split('~')[0].strip()}\n"
            f"No. Observations:            {self.nobs}\n"
            f"No. Entities:                {self.n_entities}\n"
            f"No. Time periods:            {self.n_time}\n"
            f"Covariance Type:             {self.cov_type}\n"
            f"R-squared (overall):       {self.r_squared_overall:.6f}\n"
            f"R-squared (within):        {self.r_squared_within:.6f}\n"
            f"R-squared (between):       {self.r_squared_between:.6f}\n"
            f"Var(Effects) / sigma2_u:     {self.sigma2_effects:.4f}\n"
            f"Var(Residual) / sigma2_e:    {self.sigma2_eps:.4f}\n"
            f"rho (share of var):          {self.rho:.4f}\n"
            f"Log-Likelihood:              {llf_str}\n"
            f"AIC:                         {aic_str}\n"
            f"BIC:                         {bic_str}\n"
            f"======================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return header + tbl + "\n======================================================================\n"

    def vcov(self) -> pd.DataFrame:
        if self._lm_fit is None:
            raise RuntimeError(
                "vcov() requires a fitted linearmodels result. "
                "This should not happen with the standard RE API."
            )
        cov = self._lm_fit.cov
        if hasattr(cov, "loc"):
            cov = cov.loc[self.coefficients.index, self.coefficients.index]
        return pd.DataFrame(
            np.asarray(cov, dtype=float),
            index=self.coefficients.index,
            columns=self.coefficients.index,
        )

    def predict(self, newdata: pd.DataFrame | None = None) -> pd.Series:
        if newdata is None:
            return self.fitted_values
        from formulaic import Formula
        matrices = Formula(self.rhs_formula).get_model_matrix(newdata, na_action="drop")
        XX = matrices.rhs if hasattr(matrices, "rhs") else matrices
        return pd.Series(
            np.dot(XX.values, self.coefficients.values),
            index=XX.index,
            name="predicted",
        )

    def _isnan(self, v: float) -> bool:
        try:
            return np.isnan(v)
        except (TypeError, ValueError):
            return True


class HausmanResult(BaseModel):
    """Result of a Hausman test comparing fixed-effects vs random-effects estimates."""

    def __init__(
        self,
        *,
        formula: str,
        statistic: float,
        p_value: float,
        df: int,
        coef_diff: pd.Series,
        var_diff: pd.Series,
        common_terms: list[str],
        call: dict[str, Any],
        alpha: float = 0.05,
    ) -> None:
        self.formula = formula
        self.data_shape = (len(common_terms), 1)
        self.cov_type = "hausman"
        self.call = call
        self.timestamp = datetime.now()
        self.package_version = __version__

        self.statistic = statistic
        self.p_value = p_value
        self.df = df
        self.coef_diff = coef_diff
        self.var_diff = var_diff
        self.common_terms = common_terms
        self.alpha = alpha

        self._freeze()

    def rejected_at(self, alpha: float | None = None) -> bool:
        """True if the RE estimator is rejected at significance ``alpha``."""
        if alpha is None:
            alpha = self.alpha
        if self.df <= 0 or not np.isfinite(self.statistic):
            return False
        crit = _chi2.ppf(1.0 - alpha, self.df)
        return bool(self.statistic > crit)

    def tidy(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "term": ["statistic", "p_value", "df", "rejected"],
            "value": [
                self.statistic,
                self.p_value,
                self.df,
                self.rejected_at(),
            ],
        })
        df.index.name = None
        return df

    def summary(self) -> str:
        decision = "REJECT H0 (RE inconsistent)" if self.rejected_at() else "fail to reject H0 (RE consistent)"
        header = (
            f"                        Hausman Test (FE vs RE)                        \n"
            f"======================================================================\n"
            f"Formula:                    {self.formula}\n"
            f"Compared terms:             {', '.join(self.common_terms)}\n"
            f"H statistic:                {self.statistic:.6f}\n"
            f"Degrees of freedom:         {self.df}\n"
            f"P-value:                    {self.p_value:.6e}\n"
            f"Decision (alpha={self.alpha}):       {decision}\n"
            f"======================================================================\n"
        )
        return header

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["statistic"] = self.statistic
        d["p_value"] = self.p_value
        d["df"] = self.df
        d["rejected_at_default_alpha"] = self.rejected_at()
        d["coef_diff"] = self.coef_diff.to_dict()
        return d


class FirstDifferenceResult(OLSResult):
    """Result of a first-difference panel estimator.

    Identical to :class:`OLSResult` (first-differenced OLS *is* OLS on the
    transformed data) but tagged with ``method = "first-difference"``.
    """

    def __init__(self, *, method: str = "first-difference", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # super().__init__ freezes the instance; bypass immutability for the tag.
        object.__setattr__(self, "method", method)


def _panel_ols_result(
    result_cls: type,
    formula: str,
    rhs_formula: str,
    fit: Any,
    cov_type: str,
    call: dict[str, Any],
) -> Any:
    """Build an OLS-like result from a linearmodels panel fit."""
    params = fit.params
    se = fit.std_errors
    tz = fit.tstats
    pv = fit.pvalues
    ci = fit.conf_int()

    coef = pd.Series(params.values, index=params.index)
    std = pd.Series(se.values, index=params.index)
    t = pd.Series(tz.values, index=params.index)
    p = pd.Series(pv.values, index=params.index)
    conf_int = pd.DataFrame(
        {"lower": ci.iloc[:, 0].values, "upper": ci.iloc[:, 1].values},
        index=params.index,
    )

    n = int(fit.nobs)
    k = int(fit.df_model)
    df_resid = int(fit.df_resid)

    resid = pd.Series(
        np.asarray(fit.resids, dtype=float).ravel(),
        index=getattr(fit.resids, "index", None),
        name="residuals",
    )
    # linearmodels FirstDifference returns fitted_values for the full sample
    # (including the dropped first period per entity) but resids only for the
    # differenced sample; align fitted to the residuals' index so diagnostics
    # (and fitted/residual lengths) stay consistent.
    try:
        fv_series = fit.fitted_values.iloc[:, 0] if hasattr(fit.fitted_values, "iloc") \
            else pd.Series(np.asarray(fit.fitted_values, dtype=float).ravel())
        if resid.index is not None and len(fv_series) != len(resid):
            fv_series = fv_series.reindex(resid.index)
        fitted = pd.Series(np.asarray(fv_series, dtype=float).ravel(), name="fitted")
    except Exception:
        fitted = pd.Series(np.asarray(fit.fitted_values, dtype=float).ravel(), name="fitted")

    r2 = float(fit.rsquared)
    sse = float(np.sum(resid.values ** 2))
    rsd = float(np.sqrt(sse / max(df_resid, 1)))
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(df_resid, 1) if df_resid > 0 else float("nan")

    if k > 0 and (1.0 - r2) > 1e-15 and df_resid > 0:
        f_stat = (r2 / k) / ((1.0 - r2) / df_resid)
        f_p = float(1.0 - _f.cdf(f_stat, k, df_resid))
    else:
        f_stat = float("nan")
        f_p = float("nan")

    if sse > 0 and n > 0:
        llf = float(-0.5 * n * (1.0 + np.log(2.0 * np.pi * sse / n)))
    else:
        llf = float("nan")
    aic = float(2.0 * k - 2.0 * llf) if np.isfinite(llf) else float("nan")
    bic = float(k * np.log(n) - 2.0 * llf) if np.isfinite(llf) else float("nan")

    cond = 0.0
    try:
        X = np.asarray(fit.model.exog, dtype=float)
        if X.shape[1] > 0:
            cond = float(np.linalg.cond(X))
    except Exception:
        cond = 0.0

    try:
        X_full = pd.DataFrame(np.asarray(fit.model.exog, dtype=float), columns=params.index)
    except Exception:
        X_full = None

    cov_df = None
    try:
        lm_cov = fit.cov
        if hasattr(lm_cov, "loc"):
            cov_df = lm_cov.loc[params.index, params.index]
        else:
            cov_df = pd.DataFrame(
                np.asarray(lm_cov, dtype=float), index=params.index, columns=params.index,
            )
    except Exception:
        cov_df = None

    result = result_cls(
        formula=formula,
        rhs_formula=rhs_formula,
        nobs=n,
        df_resid=df_resid,
        df_model=k,
        cov_type=cov_type,
        coefficients=coef,
        std_errors=std,
        t_stats=t,
        p_values=p,
        conf_int=conf_int,
        r_squared=r2,
        adj_r_squared=adj_r2,
        f_statistic=f_stat,
        f_p_value=f_p,
        rsd=rsd,
        llf=llf,
        aic=aic,
        bic=bic,
        fitted=fitted,
        residuals=resid,
        call=call,
        condition_number=cond,
        _X=X_full,
        _sm_fit=None,
    )
    # linearmodels-built results have no statsmodels _sm_fit; store the
    # covariance directly so vcov() works for FD / Driscoll-Kraay results.
    if cov_df is not None:
        object.__setattr__(result, "_cov", cov_df)
    return result


def _re_result_from_fit(
    formula: str,
    rhs_formula: str,
    fit: Any,
    cov_type: str,
    call: dict[str, Any],
) -> RandomEffectsResult:
    params = fit.params
    se = fit.std_errors
    tz = fit.tstats
    pv = fit.pvalues
    ci = fit.conf_int()

    coef = pd.Series(params.values, index=params.index)
    std = pd.Series(se.values, index=params.index)
    z = pd.Series(tz.values, index=params.index)
    p = pd.Series(pv.values, index=params.index)
    conf_int = pd.DataFrame(
        {"lower": ci.iloc[:, 0].values, "upper": ci.iloc[:, 1].values},
        index=params.index,
    )

    n = int(fit.nobs)
    resid = pd.Series(np.asarray(fit.resids, dtype=float).ravel(), name="residuals")
    fitted = pd.Series(np.asarray(fit.fitted_values, dtype=float).ravel(), name="fitted")

    theta = fit.theta
    if hasattr(theta, "columns"):
        theta_series = pd.Series(theta["theta"].values, index=theta.index)
    else:
        theta_series = pd.Series(np.asarray(theta, dtype=float).ravel())

    n_entities = int(len(theta_series))
    n_time = int(round(n / n_entities)) if n_entities > 0 else 0

    sigma2_effects = float(getattr(fit, "_sigma2_effects", float("nan")))
    sigma2_eps = float(getattr(fit, "_sigma2_eps", float("nan")))
    rho = float(getattr(fit, "_rho", float("nan")))

    return RandomEffectsResult(
        formula=formula,
        rhs_formula=rhs_formula,
        nobs=n,
        n_entities=n_entities,
        n_time=n_time,
        coefficients=coef,
        std_errors=std,
        z_stats=z,
        p_values=p,
        conf_int=conf_int,
        r_squared=float(fit.rsquared),
        r_squared_within=float(getattr(fit, "rsquared_within", float("nan"))),
        r_squared_between=float(getattr(fit, "rsquared_between", float("nan"))),
        r_squared_overall=float(getattr(fit, "rsquared_overall", float("nan"))),
        cov_type=cov_type,
        theta=theta_series,
        sigma2_effects=sigma2_effects,
        sigma2_eps=sigma2_eps,
        rho=rho,
        llf=float("nan"),
        aic=float("nan"),
        bic=float("nan"),
        fitted=fitted,
        residuals=resid,
        call=call,
        _lm_fit=fit,
    )


def _hausman_test(
    fe_result: Any,
    re_result: Any,
    alpha: float = 0.05,
) -> HausmanResult:
    """Hausman (1978) test: H = (b_fe - b_re)' (V_fe - V_re)^+ (b_fe - b_re)."""
    fe_coefs = fe_result.coefficients
    re_coefs = re_result.coefficients
    common = sorted(set(fe_coefs.index) & set(re_coefs.index))
    if not common:
        raise ValueError(
            "No common coefficients between the FE and RE results to compare. "
            "Ensure both were estimated on the same covariates."
        )

    b_fe = fe_coefs[common].values.astype(float)
    b_re = re_coefs[common].values.astype(float)
    V_fe = fe_result.vcov().loc[common, common].values.astype(float)
    V_re = re_result.vcov().loc[common, common].values.astype(float)

    d = b_fe - b_re
    V = V_fe - V_re
    V = (V + V.T) / 2.0  # symmetrize against floating-point noise
    V_inv = np.linalg.pinv(V)

    H = float(d @ V_inv @ d)
    k = len(common)
    # The Hausman quadratic form is non-negative by construction; a tiny
    # negative value is pure finite-sample noise when V_fe - V_re is not
    # numerically positive semi-definite. Clamp to keep the statistic monotone.
    H = max(H, 0.0)
    if k == 0 or not np.isfinite(H):
        p_value = float("nan")
    else:
        p_value = float(1.0 - _chi2.cdf(H, k))

    coef_diff = pd.Series(d, index=common, name="coef_diff")
    var_diff = pd.Series(np.diag(V), index=common, name="var_diff")

    call = {
        "alpha": alpha,
        "common_terms": common,
        "timestamp": str(datetime.now()),
        "package_version": __version__,
    }

    return HausmanResult(
        formula=getattr(fe_result, "formula", getattr(re_result, "formula", "")),
        statistic=H,
        p_value=p_value,
        df=k,
        coef_diff=coef_diff,
        var_diff=var_diff,
        common_terms=common,
        call=call,
        alpha=alpha,
    )
