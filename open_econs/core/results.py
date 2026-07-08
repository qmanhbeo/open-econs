from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from open_econs.core.base import BaseModel


class OLSResult(BaseModel):
    """Result of an ordinary least-squares regression.

    All numeric arrays are stored as ``pd.Series`` / ``pd.DataFrame``
    indexed by variable name.
    """

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
        t_stats: pd.Series,
        p_values: pd.Series,
        conf_int: pd.DataFrame,
        r_squared: float,
        adj_r_squared: float,
        f_statistic: float,
        f_p_value: float,
        rsd: float,
        fitted: pd.Series,
        residuals: pd.Series,
        call: dict[str, Any],
    ) -> None:
        self.formula = formula
        self.data_shape = (nobs, coefficients.shape[0])
        self.cov_type = cov_type
        self.call = call
        self.timestamp = datetime.now()
        self.package_version = "0.1.0"

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
        self.fitted_values = fitted if fitted is not None else pd.Series(dtype=float)
        self.residuals = residuals

        self._freeze()

    def tidy(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "Variable": self.coefficients.index,
            "Coef": self.coefficients.values,
            "Std Err": self.std_errors.values,
            "t": self.t_stats.values,
            "P>|t|": self.p_values.values,
            "[0.025": self.conf_int["lower"].values,
            "0.975]": self.conf_int["upper"].values,
        })
        df.index.name = None
        return df

    def summary(self) -> str:
        header = (
            f"                            OLS Regression Results                            \n"
            f"======================================================================\n"
            f"Dep. Variable:               {self.formula.split('~')[0].strip()}\n"
            f"No. Observations:            {self.nobs}\n"
            f"Df Residuals:                {self.df_resid}\n"
            f"Df Model:                    {self.df_model}\n"
            f"Covariance Type:             {self.cov_type}\n"
            f"R-squared:                   {self.r_squared:.6f}\n"
            f"Adj. R-squared:              {self.adj_r_squared:.6f}\n"
            f"F-statistic:                 {self.f_statistic:.4f}\n"
            f"Prob (F-statistic):          {self.f_p_value:.6e}\n"
            f"Log-Likelihood:              N/A\n"
            f"AIC:                         N/A\n"
            f"BIC:                         N/A\n"
            f"======================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return header + tbl + "\n======================================================================\n"

    def predict(self, newdata: pd.DataFrame | None = None) -> pd.Series:
        from formulaic import Formula

        if newdata is None:
            return self.fitted_values
        XX = Formula(self.formula).get_model_matrix(newdata, na_action="drop").rhs
        pred = pd.Series(
            np.dot(XX.values, self.coefficients.values),
            index=XX.index,
            name="predicted",
        )
        return pred


class OaxacaResult(BaseModel):
    """Result of an Oaxaca-Blinder decomposition.

    This is **not** a subclass of ``OLSResult`` — decompositions have
    different semantics and do not support predict().
    """

    def __init__(
        self,
        *,
        formula: str,
        nobs: int,
        cov_type: str,
        explained: float,
        unexplained: float,
        total_gap: float,
        decomposition_type: str,
        by_groups: tuple[str, str],
        std: pd.Series | None,
        call: dict[str, Any],
    ) -> None:
        self.formula = formula
        self.data_shape = (nobs, 0)
        self.cov_type = cov_type
        self.call = call
        self.timestamp = datetime.now()
        self.package_version = "0.1.0"

        self.nobs = nobs
        self.explained = explained
        self.unexplained = unexplained
        self.total_gap = total_gap
        self.type = decomposition_type
        self.by_groups = by_groups
        self.std = std

        self._freeze()

    def tidy(self) -> pd.DataFrame:
        if self.type == "two-fold":
            data = {
                "Component": ["Explained", "Unexplained", "Total Gap"],
                "Effect": [self.explained, self.unexplained, self.total_gap],
            }
        else:
            data = {
                "Component": ["Endowment", "Coefficients", "Interaction", "Total Gap"],
                "Effect": [self.explained, self.unexplained, 0.0, self.total_gap],
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