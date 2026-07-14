from __future__ import annotations

import pandas as pd

from open_econs.core.base import BaseModel


class UnitRootResult(BaseModel):
    """Result of a unit-root / stationarity test (ADF, PP, KPSS, DF-GLS, ZA).

    The displayed critical values come from the *backend's native* table
    (e.g. MacKinnon 2010 for ADF, Hobijn 2004 for KPSS).  The exact vintage is
    always surfaced in :attr:`cv_vintage` and in :meth:`summary` so a user
    comparing against a Stata/R printout is never silently misled -- the
    underlying asymptotic test is the same test, only the finite-sample table
    differs, and that difference is labelled rather than hidden (standing rule
    2).  The MacKinnon (1994) p-value (:attr:`pvalue`) is the one point of
    genuine agreement across ``arch`` and Stata and is the project's primary
    parity anchor.
    """

    def __init__(
        self,
        *,
        test_name: str,
        stat: float,
        pvalue: float | None,
        critical_values: dict[str, float],
        lags: int,
        trend: str,
        nobs: int,
        cv_vintage: str,
        null_hypothesis: str,
        alternative_hypothesis: str,
        call: dict,
    ) -> None:
        self.test_name = test_name
        self.stat = float(stat)
        self.pvalue = None if pvalue is None else float(pvalue)
        self.critical_values = {k: float(v) for k, v in critical_values.items()}
        self.lags = int(lags)
        self.trend = trend
        self.nobs = int(nobs)
        self.cv_vintage = cv_vintage
        self.null_hypothesis = null_hypothesis
        self.alternative_hypothesis = alternative_hypothesis
        self.call = call
        self._freeze()

    def tidy(self) -> pd.DataFrame:
        """Return the key test quantities as a one-row broom-style table."""
        row = {
            "test": self.test_name,
            "statistic": self.stat,
            "pvalue": self.pvalue,
            "lags": self.lags,
            "trend": self.trend,
            "nobs": self.nobs,
            "cv_1pct": self.critical_values.get("1%"),
            "cv_5pct": self.critical_values.get("5%"),
            "cv_10pct": self.critical_values.get("10%"),
            "cv_vintage": self.cv_vintage,
        }
        return pd.DataFrame([row])

    def summary(self) -> str:
        lines = [
            self.test_name,
            "=" * len(self.test_name),
            f"Test statistic : {self.stat:.6f}",
            f"p-value        : {self.pvalue if self.pvalue is None else f'{self.pvalue:.6f}'}",
            f"Lags           : {self.lags}",
            f"Trend          : {self.trend}",
            f"Observations   : {self.nobs}",
            "",
            "Critical values "
            f"[{self.cv_vintage}]:",
            f"  1% : {self.critical_values.get('1%')}",
            f"  5% : {self.critical_values.get('5%')}",
            f"  10%: {self.critical_values.get('10%')}",
            "",
            f"H0: {self.null_hypothesis}",
            f"Ha: {self.alternative_hypothesis}",
        ]
        return "\n".join(lines)


class GARCHResult(BaseModel):
    """Result of a GARCH-family volatility model estimated via ``arch_model``.

    Exposes the parameter table (coefficients, SEs, t/z, p, CI), information
    criteria, log-likelihood, residuals and conditional volatility.  All three
    reference tools (``arch``, Stata ``arch`` + ``garch()``, ``rugarch``) freely
    estimate the variance constant ``omega`` via full MLE and default to a Normal
    distribution, so no variance-targeting divergence is reconciled here (the
    GARCH lag order is always passed explicitly to stay explicit against
    Stata's requirement).
    """

    def __init__(
        self,
        *,
        params: pd.Series,
        std_errors: pd.Series,
        t_stats: pd.Series,
        p_values: pd.Series,
        conf_int: pd.DataFrame,
        llf: float,
        aic: float,
        bic: float,
        nobs: int,
        residuals: pd.Series,
        conditional_volatility: pd.Series,
        call: dict,
    ) -> None:
        self.params = params
        self.std_errors = std_errors
        self.t_stats = t_stats
        self.p_values = p_values
        self.conf_int = conf_int
        self.llf = float(llf)
        self.aic = float(aic)
        self.bic = float(bic)
        self.nobs = int(nobs)
        self.residuals = residuals
        self.conditional_volatility = conditional_volatility
        self.call = call
        self._freeze()

    def tidy(self) -> pd.DataFrame:
        """Coefficient table, one row per model parameter."""
        df = pd.DataFrame(
            {
                "term": self.params.index,
                "estimate": self.params.values,
                "std_error": self.std_errors.values,
                "statistic": self.t_stats.values,
                "p_value": self.p_values.values,
                "ci_lower": self.conf_int.iloc[:, 0].values,
                "ci_upper": self.conf_int.iloc[:, 1].values,
            }
        )
        return df

    def summary(self) -> str:
        tbl = self.tidy().to_string(index=False)
        head = (
            f"GARCH-family volatility model (n={self.nobs})\n"
            f"Log-Likelihood: {self.llf:.6f}   AIC: {self.aic:.4f}   "
            f"BIC: {self.bic:.4f}\n"
            f"{'-' * 60}\n{tbl}"
        )
        return head


class ARIMAResult(BaseModel):
    """Result of an ARIMA / ARMA model estimated via statsmodels statespace.

    Defaults to pure ML (state-space Kalman), matching both Stata ``arima`` and
    statsmodels' native default; ``method="css-ml"`` is exposed for R
    ``stats::arima`` parity.  The AR/MA sign convention was empirically verified
    to agree with Stata and R (no sign correction required in the wrapper).
    """

    def __init__(
        self,
        *,
        params: pd.Series,
        std_errors: pd.Series,
        t_stats: pd.Series,
        p_values: pd.Series,
        conf_int: pd.DataFrame,
        llf: float,
        aic: float,
        bic: float,
        nobs: int,
        residuals: pd.Series,
        fitted_values: pd.Series,
        order: tuple[int, int, int],
        method: str,
        call: dict,
    ) -> None:
        self.params = params
        self.std_errors = std_errors
        self.t_stats = t_stats
        self.p_values = p_values
        self.conf_int = conf_int
        self.llf = float(llf)
        self.aic = float(aic)
        self.bic = float(bic)
        self.nobs = int(nobs)
        self.residuals = residuals
        self.fitted_values = fitted_values
        self.order = order
        self.method = method
        self.call = call
        self._freeze()

    def tidy(self) -> pd.DataFrame:
        """Coefficient table, one row per ARIMA parameter."""
        df = pd.DataFrame(
            {
                "term": self.params.index,
                "estimate": self.params.values,
                "std_error": self.std_errors.values,
                "statistic": self.t_stats.values,
                "p_value": self.p_values.values,
                "ci_lower": self.conf_int.iloc[:, 0].values,
                "ci_upper": self.conf_int.iloc[:, 1].values,
            }
        )
        return df

    def summary(self) -> str:
        tbl = self.tidy().to_string(index=False)
        head = (
            f"ARIMA{self.order} (method={self.method}, n={self.nobs})\n"
            f"Log-Likelihood: {self.llf:.6f}   AIC: {self.aic:.4f}   "
            f"BIC: {self.bic:.4f}\n"
            f"{'-' * 60}\n{tbl}"
        )
        return head
