import warnings
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
        warnings.warn(
            "OLSResult.plot() is deprecated in v0.8 and will be removed in v0.9. "
            "This method is a generic R plot(lm) replica that does not use OE's "
            "own diagnostics (Jarque-Bera, Breusch-Pagan, Durbin-Watson, Ramsey "
            "RESET). Use self.diagnostics() for the actual test statistics and "
            "matplotlib (via pip install open-econs[plot]) for residual visuals.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        ax4.text(0.5, 0.5, "Leverage plot: planned for v0.4 — see DeprecationWarning above", ha="center", va="center", transform=ax4.transAxes)
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


class MultinomialResult(BaseModel):
    """Result of a multinomial logit (MNLogit) regression.

    Immutable result with the uniform interface (``.tidy()``, ``.summary()``,
    ``.export()``, ``.vcov()``, ``.to_latex()`` / ``.to_html()``).  Adds
    ``.margins()`` (average marginal effects per outcome category) and
    ``.predict()`` (fitted per-category probabilities).

    Shape conventions (deliberately different from the binary ``BinaryResult``):

    * ``coefficients`` / ``std_errors`` / ``z_stats`` / ``p_values`` are
      ``(category, variable)`` :class:`~pandas.DataFrame` objects: rows are the
      **non-baseline** outcome categories, columns are the regressors.  This is
      the natural layout because a multinomial logit is one logistic regression
      *per non-baseline outcome* (exactly how Stata prints ``mlogit`` equation
      blocks), and it keeps each outcome's coefficients inspectable on their
      own.
    * ``conf_int`` is a :class:`~pandas.DataFrame` with a ``(category, variable)``
      :class:`~pandas.MultiIndex` and ``lower`` / ``upper`` columns.
    * ``.vcov()`` returns the full ``(p*(K-1), p*(K-1))`` covariance matrix with a
      ``(category, variable)`` :class:`~pandas.MultiIndex` (category-major,
      matching ``coefficients.values.flatten()``).
    * ``.predict()`` returns an ``(n, K)`` :class:`~pandas.DataFrame` of per-category
      probabilities (columns = all outcome labels).
    * ``.margins()`` returns a ``dict`` keyed by outcome-category label, each
      value a tidy ``pd.DataFrame`` of average marginal effects for that outcome
      (the baseline category's AME is the negative row-sum of the others — a real
      identity of multinomial AMEs, verified numerically at fit time).
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
        categories: list,
        base_category: Any,
        non_base_categories: list,
        variable_names: list[str],
        coefficients: pd.DataFrame,
        std_errors: pd.DataFrame,
        z_stats: pd.DataFrame,
        p_values: pd.DataFrame,
        conf_int: pd.DataFrame,
        llf: float,
        aic: float,
        bic: float,
        pseudo_r2: float,
        fitted: pd.DataFrame,
        call: dict[str, Any],
        _fit: Any = None,
    ) -> None:
        self.formula = formula
        self.rhs_formula = rhs_formula
        self.data_shape = (nobs, coefficients.size)
        self.cov_type = cov_type
        self.call = call
        self.timestamp = datetime.now()
        self.package_version = __version__

        self.nobs = nobs
        self.df_resid = df_resid
        self.df_model = df_model
        self.categories = list(categories)
        self.base_category = base_category
        self.non_base_categories = list(non_base_categories)
        self.variable_names = list(variable_names)
        self.coefficients = coefficients
        self.std_errors = std_errors
        self.z_stats = z_stats
        self.p_values = p_values
        self.conf_int = conf_int
        self.llf = llf
        self.aic = aic
        self.bic = bic
        self.pseudo_r2 = pseudo_r2
        self.fitted_values = fitted if fitted is not None else pd.DataFrame()
        self._fit = _fit

        self._freeze()

    # ── helpers ──────────────────────────────────────────────────────

    def _tidy_frame(
        self,
        coef: pd.DataFrame,
        se: pd.DataFrame,
        z: pd.DataFrame,
        p: pd.DataFrame,
        ci: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []
        for cat in coef.index:
            for var in coef.columns:
                rows.append(
                    {
                        "Outcome": cat,
                        "Variable": var,
                        "Coef": coef.loc[cat, var],
                        "Std Err": se.loc[cat, var],
                        "z": z.loc[cat, var],
                        "P>|z|": p.loc[cat, var],
                        "0.025": ci.loc[(cat, var), "lower"],
                        "0.975": ci.loc[(cat, var), "upper"],
                    }
                )
        df = pd.DataFrame(rows)
        df.index.name = None
        return df

    # ── abstract interface ───────────────────────────────────────────

    def tidy(self) -> pd.DataFrame:
        return self._tidy_frame(
            self.coefficients, self.std_errors, self.z_stats,
            self.p_values, self.conf_int,
        )

    def summary(self) -> str:
        llf_str = f"{self.llf:.3f}" if self.llf is not None and not self._isnan(self.llf) else "N/A"
        aic_str = f"{self.aic:.2f}" if self.aic is not None and not self._isnan(self.aic) else "N/A"
        bic_str = f"{self.bic:.2f}" if self.bic is not None and not self._isnan(self.bic) else "N/A"
        header = (
            f"                  Multinomial Logit Regression Results                  \n"
            f"======================================================================\n"
            f"Dep. Variable:               {self.formula.split('~')[0].strip()}\n"
            f"Model:                       MNLogit (multinomial logit)\n"
            f"No. Observations:            {self.nobs}\n"
            f"Df Residuals:                {self.df_resid}\n"
            f"Df Model:                    {self.df_model}\n"
            f"Base Outcome:                {self.base_category}\n"
            f"Outcomes (non-base):         {', '.join(str(c) for c in self.non_base_categories)}\n"
            f"Pseudo R-squared:          {self.pseudo_r2:.6f}\n"
            f"Log-Likelihood:              {llf_str}\n"
            f"AIC:                         {aic_str}\n"
            f"BIC:                         {bic_str}\n"
            f"Covariance Type:             {self.cov_type}\n"
            f"======================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return header + tbl + "\n======================================================================\n"

    # ── optional stubs (reused) ──────────────────────────────────────

    def vcov(self) -> pd.DataFrame:
        if self._fit is None:
            raise RuntimeError("vcov() requires a fitted statsmodels result.")
        cov = np.asarray(self._fit.cov_params(), dtype=float)
        idx = pd.MultiIndex.from_product(
            [self.non_base_categories, self.variable_names]
        )
        return pd.DataFrame(cov, index=idx, columns=idx)

    def predict(self, newdata: pd.DataFrame | None = None, proba: bool = True) -> pd.DataFrame:
        if newdata is None:
            if not proba:
                raise ValueError(
                    "MultinomialResult.predict() with proba=False is unsupported: "
                    "there is no single 'predicted class' Series contract; pass "
                    "proba=True to get the (n, K) probability DataFrame."
                )
            return self.fitted_values
        if self._fit is None:
            raise RuntimeError("predict(newdata=...) requires a fitted statsmodels result.")
        from formulaic import Formula
        matrices = Formula(self.rhs_formula).get_model_matrix(newdata, na_action="drop")
        XX = matrices.rhs if hasattr(matrices, "rhs") else matrices
        proba_arr = self._fit.predict(XX.values)
        return pd.DataFrame(
            proba_arr,
            index=XX.index,
            columns=[str(c) for c in self.categories],
        )

    def margins(self) -> dict:
        """Average marginal effects per outcome category.

        Returns a ``dict`` mapping each outcome-category label to a tidy
        ``pd.DataFrame`` (columns: ``Variable``, ``dy/dx``, ``Std Err``, ``z``,
        ``P>|z|``, ``0.025``, ``0.975``).  The baseline category's AME is **also
        included** (it is part of ``statsmodels`` ``get_margeff``'s output, which
        returns a ``(n_regressors, K)`` array — regressors (minus the constant) by
        *all* outcome categories, including the base).

        .. note::
           This method deliberately returns a ``dict`` of DataFrames rather than
           the library's usual single tidy DataFrame.  This is an intentional
           exception to the project's single-DataFrame convention, made because
           multinomial AMEs are inherently structured **per outcome category**:
           there are ``K`` separate outcome-specific effect sets (one per
           category), not a single shared set of effects.  Returning a dict keeps
           each outcome's AMEs keyed by category label and avoids collapsing ``K``
           distinct effect vectors into an ambiguous one-frame layout.

        As an internal consistency check, the baseline category's AME equals the
        negative sum of the other categories' AMEs — a real identity of
        multinomial AMEs (they sum to zero across categories for each regressor).
        This identity is asserted in the parity tests rather than assumed.

        The underlying engine is ``statsmodels`` ``MNLogitResults.get_margeff``,
        which propagates the fit's own covariance type (robust / clustered).

        .. warning::
           ``.margins()`` has only been validated for models with a constant
           (intercept) and plain numeric regressors.  It is **not yet validated
           for categorical-regressor expansion or no-intercept specifications**;
           ``get_margeff``'s regressor layout can change shape in those cases and
           the shape assertion below will fire rather than silently misaligning
           columns.
        """
        if self._fit is None:
            raise RuntimeError("margins() requires a fitted statsmodels result.")

        me = self._fit.get_margeff(at="overall")
        margeff = np.asarray(me.margeff, dtype=float)        # (n_reg, K) reg x outcome
        margeff_se = np.asarray(me.margeff_se, dtype=float)  # (n_reg, K)
        margeff_t = np.asarray(me.tvalues, dtype=float)      # (n_reg, K)
        margeff_p = np.asarray(me.pvalues, dtype=float)      # (n_reg, K)
        margeff_ci = np.asarray(me.conf_int(), dtype=float)  # (n_reg, 2, K)

        # get_margeff drops the constant; its regressor order matches the model's
        # exog names minus the intercept.
        reg_names = [v for v in self.variable_names if v != "Intercept"]
        if len(reg_names) == 0:
            reg_names = list(self.variable_names)
        K = len(self.categories)
        expected_shape = (len(reg_names), K)
        if margeff.shape != expected_shape:
            raise RuntimeError(
                f"get_margeff shape {margeff.shape} does not match expected "
                f"regressor/category count {expected_shape} (regressors = "
                f"{len(reg_names)} exog minus intercept, categories = {K}) — "
                f"statsmodels version mismatch or unsupported model structure "
                f"(e.g. categorical-regressor expansion or no-intercept "
                f"specification), which margins() does not yet validate."
            )

        result: dict = {}
        for k in range(K):
            cat = self.categories[k]
            rows = []
            for j, v in enumerate(reg_names):
                rows.append(
                    {
                        "Variable": v,
                        "dy/dx": float(margeff[j, k]),
                        "Std Err": float(margeff_se[j, k]),
                        "z": float(margeff_t[j, k]),
                        "P>|z|": float(margeff_p[j, k]),
                        "0.025": float(margeff_ci[j, 0, k]),
                        "0.975": float(margeff_ci[j, 1, k]),
                    }
                )
            result[cat] = pd.DataFrame(rows)
        return result

    def _isnan(self, v: float) -> bool:
        try:
            return np.isnan(v)
        except (TypeError, ValueError):
            return True


class SynthResult(BaseModel):
    """Result of a synthetic control (Abadie-Diamond-Hainmueller) estimation.

    Immutable result with the uniform interface (``.tidy()``, ``.summary()``,
    ``.export()``, immutability). This is the **core point estimator only**:
    it fits the donor weights ``W`` and predictor weights ``V`` and reports the
    synthetic-counterfactual gap path. Placebo-in-space / placebo-in-time
    inference, ``plot()``, and ``predict()`` are intentionally out of scope for
    this pass and raise ``NotImplementedError`` if called.

    Shape conventions (all public outputs are named pandas objects, never raw
    numpy arrays):

    * ``weights`` — ``pd.Series`` of fitted donor weights, indexed by donor
      unit id (the coefficient-analog of the estimator).
    * ``predictor_weights`` — ``pd.Series`` of fitted predictor weights ``V``,
      indexed by predictor name (one per predictor used in the optimization).
    * ``pre_mspe`` / ``post_mspe`` — pre- and post-treatment mean squared
      prediction error of the outcome (the ADH fit-quality diagnostics).
    * ``gap_path`` — ``pd.DataFrame`` indexed by time period, spanning both pre
      and post periods, with columns ``treated``, ``synthetic``, and ``gap``.
      Exposed so the later placebo pass can reuse the counterfactual directly.

    Convergence diagnostics are surfaced straight from the nested
    :func:`scipy.optimize.minimize` (SLSQP) ``OptimizeResult`` objects — ``V``
    (outer predictor-weight loop) and ``W`` (inner donor-weight QP) — with no
    invented fields.
    """

    def __init__(
        self,
        *,
        formula: str,
        outcome: str,
        treated_unit: Any,
        donor_pool: list,
        entity: str,
        time: str,
        pre_period: Any,
        post_period: Any,
        weights: pd.Series,
        predictor_weights: pd.Series,
        predictor_names: list[str],
        pre_mspe: float,
        post_mspe: float,
        gap_path: pd.DataFrame,
        n_donors: int,
        n_pre_periods: int,
        n_post_periods: int,
        v_success: bool,
        v_loss: float,
        v_nit: int,
        v_nfev: int,
        v_message: str,
        w_success: bool,
        w_loss: float,
        w_nit: int,
        w_nfev: int,
        w_message: str,
        call: dict[str, Any],
    ) -> None:
        self.formula = formula
        self.data_shape = (len(weights) + 1, len(predictor_names))
        self.cov_type = "synthetic control"
        self.call = call
        self.timestamp = datetime.now()
        self.package_version = __version__

        self.outcome = outcome
        self.treated_unit = treated_unit
        self.donor_pool = list(donor_pool)
        self.entity = entity
        self.time = time
        self.pre_period = pre_period
        self.post_period = post_period
        self.weights = weights
        self.predictor_weights = predictor_weights
        self.predictor_names = list(predictor_names)
        self.pre_mspe = float(pre_mspe)
        self.post_mspe = float(post_mspe)
        self.gap_path = gap_path
        self.n_donors = int(n_donors)
        self.n_pre_periods = int(n_pre_periods)
        self.n_post_periods = int(n_post_periods)

        # Convergence diagnostics straight from scipy.optimize.minimize.
        self.v_success = bool(v_success)
        self.v_loss = float(v_loss)
        self.v_nit = int(v_nit)
        self.v_nfev = int(v_nfev)
        self.v_message = str(v_message)
        self.w_success = bool(w_success)
        self.w_loss = float(w_loss)
        self.w_nit = int(w_nit)
        self.w_nfev = int(w_nfev)
        self.w_message = str(w_message)

        self._freeze()

    def tidy(self) -> pd.DataFrame:
        """Donor weights table, one row per donor unit (R-broom style)."""
        df = pd.DataFrame({
            "Donor": self.weights.index,
            "Weight": self.weights.values,
        })
        df.index.name = None
        return df

    def summary(self) -> str:
        header = (
            f"                 Synthetic Control (ADH) Results                       \n"
            f"======================================================================\n"
            f"Outcome:                    {self.outcome}\n"
            f"Treated unit:               {self.treated_unit}\n"
            f"Donor pool size:            {self.n_donors}\n"
            f"Entity / Time columns:      {self.entity} / {self.time}\n"
            f"Pre period (last):          {self.pre_period}  ({self.n_pre_periods} pre periods)\n"
            f"Post period (first):        {self.post_period}  ({self.n_post_periods} post periods)\n"
            f"Pre-treatment MSPE:         {self.pre_mspe:.6e}\n"
            f"Post-treatment MSPE:        {self.post_mspe:.6e}\n"
            f"----------------------------------------------------------------------\n"
        )
        w = self.weights.sort_values(ascending=False)
        top_w = w.head(10)
        wtbl = "\n".join(f"  {idx!s:<14} {val:8.4f}" for idx, val in top_w.items())
        v = self.predictor_weights.sort_values(ascending=False)
        top_v = v.head(10)
        vtbl = "\n".join(f"  {idx!s:<22} {val:8.4f}" for idx, val in top_v.items())
        conv = (
            f"V-optim converged:         {self.v_success} "
            f"(loss={self.v_loss:.6e}, nit={self.v_nit}, nfev={self.v_nfev})\n"
            f"W-solve converged:          {self.w_success} "
            f"(loss={self.w_loss:.6e}, nit={self.w_nit}, nfev={self.w_nfev})\n"
        )
        return (
            header
            + "Top donor weights:\n"
            + wtbl
            + "\n\nTop predictor weights:\n"
            + vtbl
            + "\n\n"
            + conv
            + "======================================================================\n"
        )

    def vcov(self) -> pd.DataFrame:
        raise NotImplementedError(
            "SynthResult.vcov() is not available in this pass: synthetic control "
            "inference (placebo-in-space / placebo-in-time) is a separate, later "
            "scoped task. The point-estimator result carries no covariance matrix."
        )

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["outcome"] = self.outcome
        d["treated_unit"] = self.treated_unit
        d["donor_pool"] = list(self.donor_pool)
        d["entity"] = self.entity
        d["time"] = self.time
        d["pre_period"] = self.pre_period
        d["post_period"] = self.post_period
        d["weights"] = {str(k): float(v) for k, v in self.weights.items()}
        d["predictor_weights"] = {
            str(k): float(v) for k, v in self.predictor_weights.items()
        }
        d["pre_mspe"] = self.pre_mspe
        d["post_mspe"] = self.post_mspe
        d["n_donors"] = self.n_donors
        d["n_pre_periods"] = self.n_pre_periods
        d["n_post_periods"] = self.n_post_periods
        d["gap_path"] = {
            str(t): {
                "treated": float(self.gap_path.loc[t, "treated"]),
                "synthetic": float(self.gap_path.loc[t, "synthetic"]),
                "gap": float(self.gap_path.loc[t, "gap"]),
            }
            for t in self.gap_path.index
        }
        d["convergence"] = {
            "v": {
                "success": self.v_success,
                "loss": self.v_loss,
                "nit": self.v_nit,
                "nfev": self.v_nfev,
                "message": self.v_message,
            },
            "w": {
                "success": self.w_success,
                "loss": self.w_loss,
                "nit": self.w_nit,
                "nfev": self.w_nfev,
                "message": self.w_message,
            },
        }
        return d