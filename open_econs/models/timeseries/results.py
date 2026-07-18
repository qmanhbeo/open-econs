from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from open_econs.core.base import BaseModel

if TYPE_CHECKING:
    from statsmodels.tsa.vector_ar.var_model import VARResults as _VARResults
    from statsmodels.tsa.vector_ar.vecm import VECMResults as _VECMResults


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


class VARResult(BaseModel):
    """Result of a VAR(p) model estimated via ``statsmodels.tsa.VAR``.

    Wraps ``VARResults`` and exposes coefficient matrices, residual
    covariance, information criteria, Granger causality tests, IRF, and
    FEVD.  Two IC conventions are provided: the default matches
    statsmodels/Stata-standard/R-vars (all parameters in penalty), and
    ``lutstats`` matches Stata's ``lutstats`` option (only AR parameters
    in penalty, following Lutkepohl 2005).
    """

    def __init__(
        self,
        *,
        k_ar: int,
        neqs: int,
        nobs: int,
        n_totobs: int,
        coefs: np.ndarray,
        sigma_u: np.ndarray,
        params: np.ndarray,
        llf: float,
        aic: float,
        bic: float,
        hqic: float,
        fpe: float,
        aic_lutstats: float,
        bic_lutstats: float,
        hqic_lutstats: float,
        residuals: np.ndarray,
        names: list[str],
        trend: str,
        _sm_result: _VARResults,
        call: dict,
    ) -> None:
        self.k_ar = int(k_ar)
        self.neqs = int(neqs)
        self.nobs = int(nobs)
        self.n_totobs = int(n_totobs)
        self.coefs = coefs
        self.sigma_u = sigma_u
        self.params = params
        self.llf = float(llf)
        self.aic = float(aic)
        self.bic = float(bic)
        self.hqic = float(hqic)
        self.fpe = float(fpe)
        self.aic_lutstats = float(aic_lutstats)
        self.bic_lutstats = float(bic_lutstats)
        self.hqic_lutstats = float(hqic_lutstats)
        self.residuals = residuals
        self.names = list(names)
        self.trend = trend
        self._sm_result = _sm_result
        self.call = call
        self._freeze()

    def tidy(self) -> pd.DataFrame:
        """Coefficient table: one row per equation, columns are lag coefficients."""
        rows = []
        for eq_idx, name in enumerate(self.names):
            row = {"equation": name}
            for lag in range(1, self.k_ar + 1):
                for var_idx, var_name in enumerate(self.names):
                    row["L%d.%s" % (lag, var_name)] = self.coefs[lag - 1, eq_idx, var_idx]
            rows.append(row)
        return pd.DataFrame(rows)

    def test_causality(
        self, caused: int | str | list, causing: int | str | list | None = None,
        kind: str = "f", signif: float = 0.05,
    ) -> Any:
        """Granger causality test.  See statsmodels ``VARResults.test_causality``."""
        return self._sm_result.test_causality(caused, causing, kind=kind, signif=signif)

    def test_inst_causality(
        self, causing: int | str | list, signif: float = 0.05,
    ) -> Any:
        """Instantaneous causality test.  See statsmodels ``VARResults.test_inst_causality``."""
        return self._sm_result.test_inst_causality(causing, signif=signif)

    def irf(self, periods: int = 10, **kwargs: Any) -> Any:
        """Impulse response functions."""
        return self._sm_result.irf(periods=periods, **kwargs)

    def fevd(self, periods: int = 10, **kwargs: Any) -> Any:
        """Forecast error variance decomposition."""
        return self._sm_result.fevd(periods=periods, **kwargs)

    def summary(self) -> str:
        return self._sm_result.summary()


class LagOrderResult(BaseModel):
    """Result of VAR lag-order selection.

    Provides two IC conventions: the default (all parameters in penalty,
    matching Stata ``varsoc`` / R ``VARselect`` / statsmodels) and
    ``lutstats`` (only AR parameters in penalty, matching Stata's
    ``lutstats`` option).
    """

    def __init__(
        self,
        *,
        ic_values: dict[str, list[float]],
        selected: dict[str, int],
        selected_lutstats: dict[str, int],
        ic_values_lutstats: dict[str, list[float]],
        maxlags: int,
        neqs: int,
        nobs: int,
        trend: str,
        call: dict,
    ) -> None:
        self.ic_values = ic_values
        self.selected = selected
        self.selected_lutstats = selected_lutstats
        self.ic_values_lutstats = ic_values_lutstats
        self.maxlags = maxlags
        self.neqs = neqs
        self.nobs = nobs
        self.trend = trend
        self.call = call
        self._freeze()

    def tidy(self) -> pd.DataFrame:
        """IC values across lag orders as a DataFrame."""
        rows: list[dict[str, Any]] = []
        for lag_idx in range(self.maxlags):
            row: dict[str, Any] = {"lag": lag_idx + 1}
            for ic in ["aic", "bic", "hqic", "fpe"]:
                if ic in self.ic_values and lag_idx < len(self.ic_values[ic]):
                    row[ic] = self.ic_values[ic][lag_idx]
            for ic in ["aic", "bic", "hqic"]:
                if ic in self.ic_values_lutstats and lag_idx < len(self.ic_values_lutstats[ic]):
                    row[ic + "_lut"] = self.ic_values_lutstats[ic][lag_idx]
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [
            "VAR Lag Order Selection",
            "=" * 30,
            f"Max lags  : {self.maxlags}",
            f"Variables : {self.neqs}",
            f"Trend     : {self.trend}",
            "",
            "Standard IC (all parameters in penalty):",
        ]
        for ic in ["aic", "bic", "hqic", "fpe"]:
            lines.append(f"  {ic.upper():6s} -> lag {self.selected[ic]}")
        lines.append("")
        lines.append("Lutkepohl IC (only AR parameters in penalty):")
        for ic in ["aic", "bic", "hqic"]:
            lines.append(f"  {ic.upper():6s} -> lag {self.selected_lutstats[ic]}")
        return "\n".join(lines)


class JohansenResult(BaseModel):
    """Result of Johansen cointegration test.

    Uses Osterwald-Lenum (1992) critical values by default (matching Stata
    ``vecrank`` and R ``urca::ca.jo``).  Statsmodels' native MacKinnon-Haug-
    Michelis (1996) tables are available via ``cvt_mackinnon`` /
    ``cvm_mackinnon`` for reference.
    """

    def __init__(
        self,
        *,
        trace_stat: pd.Series,
        max_eig_stat: pd.Series,
        cvt: pd.DataFrame,
        cvm: pd.DataFrame,
        cvt_mackinnon: pd.DataFrame,
        cvm_mackinnon: pd.DataFrame,
        eigvals: np.ndarray,
        neqs: int,
        k_ar_diff: int,
        det_order: int,
        nobs: int,
        trace_stat_rank: int,
        max_eig_stat_rank: int,
        call: dict,
    ) -> None:
        self.trace_stat = trace_stat
        self.max_eig_stat = max_eig_stat
        self.cvt = cvt
        self.cvm = cvm
        self.cvt_mackinnon = cvt_mackinnon
        self.cvm_mackinnon = cvm_mackinnon
        self.eigvals = eigvals
        self.neqs = neqs
        self.k_ar_diff = k_ar_diff
        self.det_order = det_order
        self.nobs = nobs
        self.trace_stat_rank = trace_stat_rank
        self.max_eig_stat_rank = max_eig_stat_rank
        self.call = call
        self._freeze()

    def tidy(self) -> pd.DataFrame:
        """Trace and max-eigen statistics with Osterwald-Lenum CVs."""
        df = pd.DataFrame({
            "hypothesis": self.trace_stat.index,
            "trace_stat": self.trace_stat.values,
            "trace_cv_5pct": self.cvt["5%"].values,
            "max_eig_stat": self.max_eig_stat.values,
            "max_eig_cv_5pct": self.cvm["5%"].values,
        })
        return df

    def summary(self) -> str:
        lines = [
            "Johansen Cointegration Test",
            "=" * 30,
            f"Variables    : {self.neqs}",
            f"Lag diff     : {self.k_ar_diff}",
            f"Det order    : {self.det_order}",
            f"Observations : {self.nobs}",
            "",
            "Trace test (Osterwald-Lenum 1992):",
        ]
        for i in range(self.neqs):
            lines.append(
                "  r<=%d: stat=%.4f  CV_5%%=%.4f" % (
                    i, float(self.trace_stat.iloc[i]), float(self.cvt.iloc[i, 1])
                )
            )
        lines.append(f"  Selected rank (trace): {self.trace_stat_rank}")
        lines.append("")
        lines.append("Max-eigenvalue test (Osterwald-Lenum 1992):")
        for i in range(self.neqs):
            lines.append(
                "  r<=%d: stat=%.4f  CV_5%%=%.4f" % (
                    i, float(self.max_eig_stat.iloc[i]), float(self.cvm.iloc[i, 1])
                )
            )
        lines.append(f"  Selected rank (max-eig): {self.max_eig_stat_rank}")
        return "\n".join(lines)


class GrangerResult(BaseModel):
    """Result of a Granger causality or instantaneous causality test."""

    def __init__(
        self,
        *,
        test_name: str,
        test_statistic: float,
        df: tuple[int, int] | int,
        pvalue: float,
        caused: list[str],
        causing: list[str],
        method: str,
        signif: float,
        conclusion: str,
        call: dict,
    ) -> None:
        self.test_name = test_name
        self.test_statistic = float(test_statistic)
        self.df = df
        self.pvalue = float(pvalue)
        self.caused = list(caused)
        self.causing = list(causing)
        self.method = method
        self.signif = float(signif)
        self.conclusion = conclusion
        self.call = call
        self._freeze()

    def tidy(self) -> pd.DataFrame:
        """One-row summary of the causality test."""
        row = {
            "test": self.test_name,
            "method": self.method,
            "statistic": self.test_statistic,
            "df": str(self.df),
            "pvalue": self.pvalue,
            "causing": ", ".join(self.causing),
            "caused": ", ".join(self.caused),
            "conclusion": self.conclusion,
        }
        return pd.DataFrame([row])

    def summary(self) -> str:
        lines = [
            self.test_name,
            "=" * len(self.test_name),
            f"Method        : {self.method}",
            f"Test statistic: {self.test_statistic:.6f}",
            f"df            : {self.df}",
            f"p-value       : {self.pvalue:.6f}",
            f"Caused by     : {', '.join(self.causing)}",
            f"Causes        : {', '.join(self.caused)}",
            f"Significance  : {self.signif}",
            f"Conclusion    : {self.conclusion}",
        ]
        return "\n".join(lines)


class VECMResult(BaseModel):
    """Result of a VECM estimated via ``statsmodels.tsa.VECM``.

    Wraps ``VECMResults`` and exposes alpha, beta, gamma, sigma_u, and
    the VAR representation for IRF/FEVD and Granger causality after
    VECM estimation.
    """

    def __init__(
        self,
        *,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        sigma_u: np.ndarray,
        det_coef_coint: np.ndarray,
        det_coef: np.ndarray,
        llf: float,
        nobs: int,
        neqs: int,
        k_ar: int,
        coint_rank: int,
        deterministic: str,
        residuals: np.ndarray,
        _sm_result: _VECMResults,
        call: dict,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma_u = sigma_u
        self.det_coef_coint = det_coef_coint
        self.det_coef = det_coef
        self.llf = float(llf)
        self.nobs = nobs
        self.neqs = neqs
        self.k_ar = k_ar
        self.coint_rank = coint_rank
        self.deterministic = deterministic
        self.residuals = residuals
        self._sm_result = _sm_result
        self.call = call
        self._freeze()

    def tidy(self) -> pd.DataFrame:
        """Alpha and beta matrices of the VECM."""
        rows = []
        for i in range(self.neqs):
            for j in range(self.coint_rank):
                rows.append({
                    "matrix": "alpha",
                    "row": i,
                    "col": j,
                    "value": float(self.alpha[i, j]),
                })
        for i in range(self.neqs):
            for j in range(self.coint_rank):
                rows.append({
                    "matrix": "beta",
                    "row": i,
                    "col": j,
                    "value": float(self.beta[i, j]),
                })
        return pd.DataFrame(rows)

    def var_rep(self) -> Any:
        """VAR representation coefficient matrices."""
        return self._sm_result.var_rep()

    def irf(self, periods: int = 10) -> Any:
        """Impulse response functions from the VAR representation."""
        return self._sm_result.irf(periods=periods)

    def test_normality(self, signif: float = 0.05) -> Any:
        """Multivariate normality test."""
        return self._sm_result.test_normality(signif=signif)

    def test_whiteness(self, nlags: int = 10, signif: float = 0.05, adjusted: bool = False) -> Any:
        """Residual whiteness test."""
        return self._sm_result.test_whiteness(nlags=nlags, signif=signif, adjusted=adjusted)

    def summary(self, alpha: float = 0.05) -> str:
        return self._sm_result.summary(alpha=alpha)


class ARDLResult(BaseModel):
    """Result of an ARDL(p, q1, ..., qk) model estimated via ``statsmodels``.

    Wraps ``statsmodels.tsa.ardl.ARDLResults``.  The coefficient table matches
    Stata SSC ``ardl`` and R ``ARDL::ardl`` to machine precision (all three
    fit the same OLS ARDL regression).  The PSS (2001) bounds test for a level
    relationship is reached through :meth:`bounds_test`, which reconciles the
    F-bounds (the one cross-tool agreement anchor at 1e-6) and adds the
    OE-computed t-bounds that statsmodels omits.
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
        hqic: float,
        nobs: int,
        resid: pd.Series,
        fitted_values: pd.Series,
        y_name: str,
        exog_names: list[str],
        trend: str,
        order: Any,
        lags: Any,
        _sm_result: Any,
        _is_uecm: bool,
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
        self.hqic = float(hqic)
        self.nobs = int(nobs)
        self.resid = resid
        self.fitted_values = fitted_values
        self.y_name = y_name
        self.exog_names = list(exog_names)
        self.trend = trend
        self.order = order
        self.lags = lags
        self._sm_result = _sm_result
        self._is_uecm = _is_uecm
        self.call = call
        self._freeze()

    def tidy(self) -> pd.DataFrame:
        """Coefficient table, one row per ARDL parameter."""
        return pd.DataFrame(
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

    def bounds_test(
        self,
        case: int,
        *,
        cv_vintage: str = "pss2001",
        signif: Any = (0.10, 0.05, 0.01),
    ) -> "BoundsTestResult":
        """PSS (2001) bounds test for a level relationship. See :func:`bounds_test`."""
        from open_econs.models.timeseries.ardl import bounds_test as _bt

        return _bt(self, case, cv_vintage=cv_vintage, signif=signif)

    def summary(self) -> str:
        tbl = self.tidy().to_string(index=False)
        label = "UECM" if self._is_uecm else "ARDL"
        head = (
            f"{label} ({self.y_name} ~ {', '.join(self.exog_names)}, "
            f"trend={self.trend}, n={self.nobs})\n"
            f"Log-Likelihood: {self.llf:.6f}   AIC: {self.aic:.4f}   "
            f"BIC: {self.bic:.4f}\n"
            f"{'-' * 60}\n{tbl}"
        )
        return head


class UECMResult(ARDLResult):
    """Result of the unrestricted error-correction (UECM) form of an ARDL.

    Extends :class:`ARDLResult` with the long-run (level) coefficients and the
    speed-of-adjustment (error-correction) term.  Long-run coefficients follow
    Stata ``ardl, ec`` / R ``ARDL::multipliers()`` (``LR = -theta / rho``) by
    default; ``lr_sign="statsmodels"`` returns the raw ``ci_params`` sign.
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
        hqic: float,
        nobs: int,
        resid: pd.Series,
        fitted_values: pd.Series,
        y_name: str,
        exog_names: list[str],
        trend: str,
        order: Any,
        lags: Any,
        long_run: pd.Series,
        ec_term: float,
        ec_term_se: float,
        ec_term_t: float,
        ec_term_pvalue: float,
        ec_term_name: str,
        lr_sign: str,
        _sm_result: Any,
        call: dict,
    ) -> None:
        # Set UECM-specific fields first; the parent initializer freezes last.
        self.long_run = long_run
        self.ec_term = float(ec_term)
        self.ec_term_se = float(ec_term_se)
        self.ec_term_t = float(ec_term_t)
        self.ec_term_pvalue = float(ec_term_pvalue)
        self.ec_term_name = ec_term_name
        self.lr_sign = lr_sign
        super().__init__(
            params=params,
            std_errors=std_errors,
            t_stats=t_stats,
            p_values=p_values,
            conf_int=conf_int,
            llf=llf,
            aic=aic,
            bic=bic,
            hqic=hqic,
            nobs=nobs,
            resid=resid,
            fitted_values=fitted_values,
            y_name=y_name,
            exog_names=exog_names,
            trend=trend,
            order=order,
            lags=lags,
            _sm_result=_sm_result,
            _is_uecm=True,
            call=call,
        )

    def long_run_table(self) -> pd.DataFrame:
        """Long-run (level) coefficients as a tidy table."""
        return pd.DataFrame(
            {
                "term": self.long_run.index,
                "long_run": self.long_run.values,
            }
        )

    def summary(self) -> str:
        base = super().summary()
        lr = self.long_run_table().to_string(index=False)
        ec = (
            f"\n\nError-correction (speed of adjustment): "
            f"{self.ec_term_name} = {self.ec_term:.6f} "
            f"(se={self.ec_term_se:.6f}, t={self.ec_term_t:.4f}, "
            f"p={self.ec_term_pvalue:.4f})\n"
            f"Long-run coefficients [lr_sign={self.lr_sign}]:\n{lr}"
        )
        return base + ec


class OrderSelectionResult(BaseModel):
    """Result of IC-based ARDL order selection via ``ardl_select_order``.

    Default ``ic="bic"`` matches Stata ``ardl`` and statsmodels; R
    ``ARDL::auto_ardl`` defaults to AIC, so ``ic=`` must be pinned explicitly
    for cross-tool parity.
    """

    def __init__(
        self,
        *,
        selected_ar_order: int,
        selected_dl_orders: dict[str, int],
        ic: str,
        ic_value: float,
        maxlag: int,
        maxorder: Any,
        trend: str,
        y_name: str,
        exog_names: list[str],
        _sm_selection: Any,
        call: dict,
    ) -> None:
        self.selected_ar_order = int(selected_ar_order)
        self.selected_dl_orders = dict(selected_dl_orders)
        self.ic = ic
        self.ic_value = float(ic_value)
        self.maxlag = int(maxlag)
        self.maxorder = maxorder
        self.trend = trend
        self.y_name = y_name
        self.exog_names = list(exog_names)
        self._sm_selection = _sm_selection
        self.call = call
        self._freeze()

    def tidy(self) -> pd.DataFrame:
        """Selected orders as a one-row table."""
        row: dict[str, Any] = {
            "y": self.y_name,
            "ar_order": self.selected_ar_order,
            "ic": self.ic,
            "ic_value": self.ic_value,
        }
        for name, o in self.selected_dl_orders.items():
            row[f"dl_{name}"] = o
        return pd.DataFrame([row])

    def summary(self) -> str:
        lines = [
            "ARDL Order Selection",
            "=" * 30,
            f"Criterion : {self.ic.upper()} = {self.ic_value:.6f}",
            f"Max lag   : {self.maxlag}",
            f"Trend     : {self.trend}",
            "",
            f"Selected AR order (p): {self.selected_ar_order}",
            "Selected DL orders (q):",
        ]
        for name, o in self.selected_dl_orders.items():
            lines.append(f"  {name}: {o}")
        return "\n".join(lines)


class BoundsTestResult(BaseModel):
    """Result of the Pesaran-Shin-Smith (2001) bounds test.

    Reports both the F-bounds and the t-bounds test.  The F-statistic is the
    one cross-tool agreement point (asserted at 1e-6 against Stata / R /
    statsmodels).  The t-bounds is computed by OE on the ``y_{t-1}``
    coefficient (statsmodels omits it) with restricted cases folding onto their
    unrestricted sibling (2->3, 4->5), matching Stata.  p-values are engine-
    specific and must not be compared cross-tool.
    """

    def __init__(
        self,
        *,
        case: int,
        cv_vintage: str,
        f_stat: float,
        f_crit_lower: dict[str, float],
        f_crit_upper: dict[str, float],
        f_pvalues: dict[str, float],
        t_stat: float | None,
        t_crit_lower: dict[str, float],
        t_crit_upper: dict[str, float],
        t_case: int,
        k: int,
        nobs: int,
        null: str,
        alternative: str,
        call: dict,
    ) -> None:
        self.case = int(case)
        self.cv_vintage = cv_vintage
        self.f_stat = float(f_stat)
        self.f_crit_lower = dict(f_crit_lower)
        self.f_crit_upper = dict(f_crit_upper)
        self.f_pvalues = dict(f_pvalues)
        self.t_stat = None if t_stat is None else float(t_stat)
        self.t_crit_lower = dict(t_crit_lower)
        self.t_crit_upper = dict(t_crit_upper)
        self.t_case = int(t_case)
        self.k = int(k)
        self.nobs = int(nobs)
        self.null = null
        self.alternative = alternative
        self.call = call
        self._freeze()

    def tidy(self) -> pd.DataFrame:
        """F- and t-bounds critical values across significance levels."""
        rows: list[dict[str, Any]] = []
        for lvl in self.f_crit_lower:
            rows.append(
                {
                    "test": "F",
                    "signif": lvl,
                    "statistic": self.f_stat,
                    "I(0)_lower": self.f_crit_lower.get(lvl),
                    "I(1)_upper": self.f_crit_upper.get(lvl),
                }
            )
        for lvl in self.t_crit_lower:
            rows.append(
                {
                    "test": "t",
                    "signif": lvl,
                    "statistic": self.t_stat,
                    "I(0)_lower": self.t_crit_lower.get(lvl),
                    "I(1)_upper": self.t_crit_upper.get(lvl),
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [
            "PSS (2001) Bounds Test for a Level Relationship",
            "=" * 47,
            f"Case          : {self.case}",
            f"Regressors (k): {self.k}",
            f"Observations  : {self.nobs}",
            f"CV vintage    : {self.cv_vintage}",
            "",
            f"H0: {self.null}",
            f"Ha: {self.alternative}",
            "",
            f"F-statistic: {self.f_stat:.6f}",
            "  signif   I(0)      I(1)",
        ]
        for lvl in self.f_crit_lower:
            lines.append(
                f"  {lvl:5s}  {self.f_crit_lower[lvl]:8.4f}  "
                f"{self.f_crit_upper[lvl]:8.4f}"
            )
        if self.t_stat is not None:
            lines.append("")
            lines.append(
                f"t-statistic: {self.t_stat:.6f}  (t-bounds case {self.t_case})"
            )
            lines.append("  signif   I(0)      I(1)")
            for lvl in self.t_crit_lower:
                lines.append(
                    f"  {lvl:5s}  {self.t_crit_lower[lvl]:8.4f}  "
                    f"{self.t_crit_upper[lvl]:8.4f}"
                )
        return "\n".join(lines)
