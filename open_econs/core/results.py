from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as _stats
from statsmodels.stats.diagnostic import het_breuschpagan as _het_breuschpagan
from statsmodels.stats.stattools import durbin_watson as _durbin_watson

from open_econs._version import __version__
from open_econs.core.base import BaseModel


def _ramsey_reset(
    fitted: np.ndarray, resid: np.ndarray, power: int = 3,
) -> tuple[float, float]:
    n = len(resid)
    y_hat_sq = np.column_stack([fitted ** p for p in range(2, power + 1)])
    X_aug = np.column_stack([np.ones(n), fitted, y_hat_sq])
    beta = np.linalg.lstsq(X_aug, fitted + resid, rcond=None)[0]
    pred_aug = X_aug @ beta
    resid_aug = (fitted + resid) - pred_aug
    ssr_r = np.sum(resid ** 2)
    ssr_u = np.sum(resid_aug ** 2)
    k = 1
    m = power - 1
    df_num = m
    df_den = n - k - m - 1
    if df_den <= 0 or ssr_u <= 0:
        return (float("nan"), float("nan"))
    f_stat = ((ssr_r - ssr_u) / m) / (ssr_u / df_den)
    p_val = 1.0 - _stats.f.cdf(f_stat, df_num, df_den)
    return (float(f_stat), float(p_val))


class OLSResult(BaseModel):
    def __init__(
        self,
        *,
        formula: str,
        rhs_formula: str,
        nobs: int,
        df_resid: int,
        df_model: int,
        cov_type: str,
        coefficients: pd.Series,
        std_errors: pd.Series,
        t_stats: pd.Series,
        p_values: pd.Series,
        conf_int: pd.DataFrame,
        r_squared: float,
        adj_r_squared: float,
        f_statistic: float,
        f_p_value: float,
        rsd: float,
        llf: float,
        aic: float,
        bic: float,
        fitted: pd.Series,
        residuals: pd.Series,
        call: dict[str, Any],
        model_spec: Any = None,
        condition_number: float = 0.0,
        _X: pd.DataFrame | None = None,
    ) -> None:
        self.formula = formula
        self.rhs_formula = rhs_formula
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
        self.t_stats = t_stats
        self.p_values = p_values
        self.conf_int = conf_int
        self.r_squared = r_squared
        self.adj_r_squared = adj_r_squared
        self.f_statistic = f_statistic
        self.f_p_value = f_p_value
        self.rsd = rsd
        self.llf = llf
        self.aic = aic
        self.bic = bic
        self.fitted_values = fitted if fitted is not None else pd.Series(dtype=float)
        self.residuals = residuals
        self._model_spec = model_spec
        self.condition_number = condition_number
        self._X = _X

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
        llf_str = f"{self.llf:.3f}" if self.llf is not None and not self._isnan(self.llf) else "N/A"
        aic_str = f"{self.aic:.2f}" if self.aic is not None and not self._isnan(self.aic) else "N/A"
        bic_str = f"{self.bic:.2f}" if self.bic is not None and not self._isnan(self.bic) else "N/A"
        header = (
            f"                            OLS Regression Results                            \n"
            f"======================================================================\n"
            f"Dep. Variable:               {self.formula.split('~')[0].strip()}\n"
            f"No. Observations:            {self.nobs}\n"
            f"Df Residuals:                {self.df_resid}\n"
            f"Df Model:                    {self.df_model}\n"
            f"Covariance Type:             {self.cov_type}\n"
            f"R-squared:                   {self.r_squared:.6f}\n"
            f"Adj. R-squared:           {self.adj_r_squared:.6f}\n"
            f"F-statistic:                 {self._fmt(self.f_statistic, '.4f')}\n"
            f"Prob (F-statistic):          {self._fmt(self.f_p_value, '.6e')}\n"
            f"Log-Likelihood:              {llf_str}\n"
            f"AIC:                         {aic_str}\n"
            f"BIC:                         {bic_str}\n"
            f"======================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return header + tbl + "\n======================================================================\n"

    def _isnan(self, v: float) -> bool:
        try:
            return np.isnan(v)
        except (TypeError, ValueError):
            return True

    def _fmt(self, v: float, spec: str) -> str:
        if self._isnan(v):
            return "N/A"
        return f"{v:{spec}}"

    def diagnostics(self) -> dict[str, tuple[float, float]]:
        res = self.residuals.values.ravel()
        fitted = self.fitted_values.values.ravel()
        results: dict[str, tuple[float, float]] = {}
        from scipy.stats import jarque_bera as _jb
        jb_stat, jb_p = _jb(res)
        results["jarque_bera"] = (float(jb_stat), float(jb_p))
        dw = _durbin_watson(res)
        results["durbin_watson"] = (float(dw), float("nan"))
        if self._X is not None and len(self._X) == len(res):
            X_vals = self._X.values.astype(float)
            bp_stat, bp_p, _, _ = _het_breuschpagan(res, X_vals)
            results["breusch_pagan"] = (float(bp_stat), float(bp_p))
        reset_stat, reset_p = _ramsey_reset(fitted, res, power=3)
        results["ramsey_reset"] = (float(reset_stat), float(reset_p))
        return results

    def wald_test(self, r_matrix: Any) -> Any:
        raise NotImplementedError(
            "wald_test() requires the statsmodels fitted result object, "
            "which is not currently stored on OLSResult. "
            "Use statsmodels directly for hypothesis testing in v0.2.0."
        )

    def f_test(self, r_matrix: Any) -> Any:
        raise NotImplementedError(
            "f_test() requires the statsmodels fitted result object, "
            "which is not currently stored on OLSResult. "
            "Use statsmodels directly for hypothesis testing in v0.2.0."
        )

    def predict(self, newdata: pd.DataFrame | None = None) -> pd.Series:
        if newdata is None:
            return self.fitted_values
        try:
            if self._model_spec is not None:
                matrices = self._model_spec.get_model_matrix(newdata, na_action="drop")
                if hasattr(matrices, "rhs"):
                    XX = matrices.rhs
                else:
                    XX = matrices
            else:
                from formulaic import Formula
                matrices = Formula(self.rhs_formula).get_model_matrix(newdata, na_action="drop")
                XX = matrices.rhs if hasattr(matrices, "rhs") else matrices
        except Exception as e:
            msg = str(e)
            if "not present in the dataset" in msg or "is not present" in msg:
                import re as _re
                m = _re.search(r"`(\w+)`", msg)
                bad_col = m.group(1) if m else self.rhs_formula
                from open_econs._internal.errors import missing_column_error
                raise missing_column_error(bad_col, newdata.columns.tolist()) from e
            raise
        pred = pd.Series(
            np.dot(XX.values, self.coefficients.values),
            index=XX.index,
            name="predicted",
        )
        return pred


class OaxacaResult(BaseModel):
    def __init__(
        self,
        *,
        formula: str,
        nobs: int,
        n_params: int,
        cov_type: str,
        explained: float,
        unexplained: float,
        interaction: float | None,
        total_gap: float,
        decomposition_type: str,
        by_groups: tuple[str, str],
        std: pd.Series | None,
        call: dict[str, Any],
        variable_detail: pd.DataFrame | None = None,
    ) -> None:
        self.formula = formula
        self.data_shape = (nobs, n_params)
        self.cov_type = cov_type
        self.call = call
        self.timestamp = datetime.now()
        self.package_version = __version__

        self.nobs = nobs
        self.explained = explained
        self.unexplained = unexplained
        self.interaction = interaction if interaction is not None else 0.0
        self.total_gap = total_gap
        self.type = decomposition_type
        self.by_groups = by_groups
        self.std = std
        self.variable_detail = variable_detail if variable_detail is not None else pd.DataFrame()

        self._freeze()

    def tidy(self, detail: bool = False) -> pd.DataFrame:
        if detail and not self.variable_detail.empty:
            return self.variable_detail
        if self.type == "two-fold":
            data = {
                "Component": ["Explained", "Unexplained", "Total Gap"],
                "Effect": [self.explained, self.unexplained, self.total_gap],
            }
        else:
            data = {
                "Component": ["Endowment", "Coefficients", "Interaction", "Total Gap"],
                "Effect": [self.explained, self.unexplained, self.interaction, self.total_gap],
            }
        df = pd.DataFrame(data)
        if self.std is not None:
            std_vals = list(self.std) + [float("nan")]
            df["Std Err"] = std_vals[: len(df)]
        return df

    def summary(self) -> str:
        header = (
            f"                     Oaxaca-Blinder Decomposition                        \n"
            f"======================================================================\n"
            f"Formula:                    {self.formula}\n"
            f"No. Observations:           {self.nobs}\n"
            f"Groups:                     {self.by_groups[0]} vs {self.by_groups[1]}\n"
            f"Type:                       {self.type}\n"
            f"======================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return header + tbl + "\n======================================================================\n"