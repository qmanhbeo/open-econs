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
    """Result of an OLS (or WLS / robust / clustered) linear regression.

    Immutable result with a uniform interface: ``.tidy()`` (coefficients, SEs,
    t/z-stats, p-values, CI), ``.summary()`` (text), ``.export()``
    (CSV/JSON/Pickle), ``.vcov()``, ``.to_latex()`` / ``.to_html()``, plus
    diagnostics ``.wald_test()`` / ``.f_test()`` and ``.ramsey_reset()`` for
    functional-form misspecification.
    """
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
        _fit: Any = None,
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
            f"Condition No.:               {self._fmt(self.condition_number, '.2e')}\n"
f"F-statistic ({self.cov_type}):     {self._fmt(self.f_statistic, '.4f')}\n"
f"Prob (F-statistic):          {self._fmt(self.f_p_value, '.6e')}\n"
            f"Log-Likelihood:              {llf_str}\n"
            f"AIC:                         {aic_str}\n"
            f"BIC:                         {bic_str}\n"
            f"======================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        diag = self.diagnostics()
        diag_lines = []
        try:
            jb_s, jb_p = diag.get("jarque_bera", (float("nan"), float("nan")))
            diag_lines.append(f"Jarque-Bera (chi2={jb_s:.3f}, p={jb_p:.4f})")
        except Exception:
            diag_lines.append("Jarque-Bera: N/A")
        try:
            bp_s, bp_p = diag.get("breusch_pagan", (float("nan"), float("nan")))
            diag_lines.append(f"Breusch-Pagan (LM={bp_s:.3f}, p={bp_p:.4f})")
        except Exception:
            diag_lines.append("Breusch-Pagan: N/A")
        try:
            dw = diag.get("durbin_watson", (float("nan"),))[0]
            diag_lines.append(f"Durbin-Watson:                  {dw:.4f}")
        except Exception:
            diag_lines.append("Durbin-Watson: N/A")
        try:
            rs_s, rs_p = diag.get("ramsey_reset", (float("nan"), float("nan")))
            diag_lines.append(f"Ramsey RESET (F={rs_s:.3f}, p={rs_p:.4f})")
        except Exception:
            diag_lines.append("Ramsey RESET: N/A")
        diag_str = "\n".join(diag_lines)
        return (
            header + tbl +
            "\n======================================================================\n"
            f"Diagnostics:\n{diag_str}\n"
            "======================================================================\n"
        )

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

    def vcov(self) -> pd.DataFrame:
        # An explicitly stored covariance (multi-way cluster, Newey-West HAC,
        # or a linearmodels-backed result) takes precedence over the backend
        # fit object so the reported SEs stay consistent with vcov().
        cov_df = getattr(self, "_cov", None)
        if cov_df is not None:
            common = [c for c in self.coefficients.index if c in cov_df.index]
            return cov_df.loc[common, common]
        if self._fit is not None:
            cov = np.asarray(self._fit.cov_params(), dtype=float)
            # Align to the reported coefficients. The full design matrix
            # (self._X) keeps the (absorbed) Intercept column, while the
            # reported coefficients drop it for FE fits; slice accordingly.
            if self._X is not None and self._X.shape[1] == cov.shape[0]:
                full_names = list(self._X.columns)
            elif getattr(self._fit.model, "exog_names", None):
                full_names = list(self._fit.model.exog_names)
            else:
                full_names = list(self.coefficients.index)
            full = pd.DataFrame(cov, index=full_names, columns=full_names)
            common = [c for c in self.coefficients.index if c in full.index]
            if not common:
                common = full_names
            return full.loc[common, common]
        raise RuntimeError(
            "vcov() requires a fitted statsmodels result. "
            "This should not happen with the standard API."
        )

    def wald_test(self, r_matrix: Any) -> Any:
        if self._fit is None:
            raise RuntimeError(
                "No fitted statsmodels result stored. "
                "wald_test() is only available when ols() is used directly."
            )
        return self._fit.wald_test(r_matrix)

    def f_test(self, r_matrix: Any) -> Any:
        if self._fit is None:
            raise RuntimeError(
                "No fitted statsmodels result stored. "
                "f_test() is only available when ols() is used directly."
            )
        return self._fit.f_test(r_matrix)

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

    def plot(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "plot() requires matplotlib. Install it with: "
                "pip install open-econs[plot]  or  pip install matplotlib"
            )
        res = self.residuals.values.ravel()
        fitted = self.fitted_values.values.ravel()
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        ax1.scatter(fitted, res, alpha=0.5, s=20)
        ax1.axhline(0, color="red", linestyle="--", linewidth=1)
        ax1.set_xlabel("Fitted values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs Fitted")
        from scipy.stats import probplot
        probplot(res, dist="norm", plot=ax2)
        ax2.get_lines()[0].set_marker("o")
        ax2.get_lines()[0].set_markersize(3)
        ax2.get_lines()[0].set_alpha(0.5)
        ax2.set_title("Normal Q-Q")
        sqrt_abs_res = np.sqrt(np.abs(res))
        ax3.scatter(fitted, sqrt_abs_res, alpha=0.5, s=20)
        ax3.set_xlabel("Fitted values")
        ax3.set_ylabel("Sqrt(|Residuals|)")
        ax3.set_title("Scale-Location")
        ax4.text(0.5, 0.5, "Leverage plot: planned for v0.4", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Residuals vs Leverage")
        fig.tight_layout()
        plt.show()


class BinaryResult(BaseModel):
    """Result of a binary-choice (logit or probit) regression.

    Immutable result with the uniform interface (``.tidy()``, ``.summary()``,
    ``.export()``, ``.vcov()``, ``.to_latex()`` / ``.to_html()``).  Adds
    ``.margins()`` (average marginal effects at the mean) and ``.predict()``
    (fitted probabilities).  Standard errors and z-stats are reported on the
    index (latent) scale.
    """
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
        z_stats: pd.Series,
        p_values: pd.Series,
        conf_int: pd.DataFrame,
        llf: float,
        aic: float,
        bic: float,
        pseudo_r2: float,
        fitted: pd.Series,
        call: dict[str, Any],
        model_type: str,
        _fit: Any = None,
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
        self.z_stats = z_stats
        self.p_values = p_values
        self.conf_int = conf_int
        self.llf = llf
        self.aic = aic
        self.bic = bic
        self.pseudo_r2 = pseudo_r2
        self.fitted_values = fitted if fitted is not None else pd.Series(dtype=float)
        self.model_type = model_type
        self._fit = _fit

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
            f"                 {self.model_type.upper()} Regression Results                 \n"
            f"======================================================================\n"
            f"Dep. Variable:               {self.formula.split('~')[0].strip()}\n"
            f"Model:                       {self.model_type.title()}\n"
            f"No. Observations:            {self.nobs}\n"
            f"Df Residuals:                {self.df_resid}\n"
            f"Df Model:                    {self.df_model}\n"
            f"Pseudo R-squared:          {self.pseudo_r2:.6f}\n"
            f"Log-Likelihood:              {llf_str}\n"
            f"AIC:                         {aic_str}\n"
            f"BIC:                         {bic_str}\n"
            f"======================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return (
            header + tbl +
            "\n======================================================================\n"
        )

    def margins(self) -> pd.DataFrame:
        if self._fit is None:
            raise RuntimeError(
                "margins() requires a fitted statsmodels result. "
                "This should not happen with the standard logit()/probit() API."
            )
        margeff = self._fit.get_margeff(at="overall")
        non_const_vars = [c for c in self.coefficients.index if c != "Intercept"]
        df = pd.DataFrame({
            "Variable": non_const_vars,
            "dy/dx": margeff.margeff,
            "Std Err": margeff.margeff_se,
            "z": margeff.tvalues,
            "P>|z|": margeff.pvalues,
            "0.025": margeff.conf_int()[:, 0],
            "0.975": margeff.conf_int()[:, 1],
        })
        df.index.name = None
        return df

    def vcov(self) -> pd.DataFrame:
        if self._fit is None:
            raise RuntimeError(
                "vcov() requires a fitted statsmodels result."
            )
        return pd.DataFrame(
            self._fit.cov_params(),
            index=self.coefficients.index,
            columns=self.coefficients.index,
        )

    def predict(self, newdata: pd.DataFrame | None = None, proba: bool = True) -> pd.Series:
        if newdata is None:
            return self.fitted_values if proba else pd.Series(
                (self.fitted_values >= 0.5).astype(int),
                index=self.fitted_values.index,
                name="predicted",
            )
        if self._fit is None:
            raise RuntimeError(
                "predict(newdata=...) requires a fitted statsmodels result."
            )
        from formulaic import Formula
        matrices = Formula(self.rhs_formula).get_model_matrix(newdata, na_action="drop")
        XX = matrices.rhs if hasattr(matrices, "rhs") else matrices
        probs = self._fit.predict(XX)
        if proba:
            return pd.Series(probs, index=XX.index, name="predicted_proba")
        return pd.Series((probs >= 0.5).astype(int), index=XX.index, name="predicted")

    def _isnan(self, v: float) -> bool:
        try:
            return np.isnan(v)
        except (TypeError, ValueError):
            return True


class OaxacaResult(BaseModel):
    """Result of an Oaxaca-Blinder (or Neumark) wage-gap decomposition.

    Immutable result with the uniform interface (``.tidy()``, ``.summary()``,
    ``.export()``, ``.to_latex()`` / ``.to_html()``).  Decomposes a mean
    outcome gap between two groups into an ``explained`` part (endowments) and
    an ``unexplained`` part (coefficients / treatment), with the overall total
    and gap stored as scalars.

    **Stata** ``oaxaca`` **→ open-econs naming**:

    ================== ==================== ==========================
    Stata ``e(b)``     OE attribute         Applies to
    ================== ==================== ==========================
    ``gap``            ``.total_gap``       all variants
    ``explained``      ``.explained``       two-fold
    ``unexplained``    ``.unexplained``     two-fold
    ``endowment``      ``.explained``       three-fold
    ``coefficients``   ``.unexplained``     three-fold
    ``interaction``    ``.interaction``     three-fold
    ================== ==================== ==========================

    In three-fold mode, ``.explained`` is the endowment effect and
    ``.unexplained`` is the coefficient effect — *not* the two-fold
    meaning of those terms.  ``.interaction`` is zero in two-fold mode.
    """
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