"""Quantile regression — Stata ``qreg``/``sqreg``/``bsqreg`` and R ``quantreg::rq`` parity.

Coefficients are obtained by the exact linear-programming solution of the
Koenker-Bassett check-function objective (the same vertex the Barrodale-Roberts
simplex finds), so they reproduce Stata ``qreg`` and R ``rq(method="br")`` to
machine precision.  Standard errors expose a rule-15 toggle ``se_method``:

* ``"stata"`` (default) — Stata ``qreg`` default VCE: the i.i.d. Koenker-Bassett
  sparsity sandwich ``V = s^2 * tau*(1-tau) * (X'X)^{-1}`` with the *fitted*
  sparsity estimate ``s = mean(X (b(tau+h) - b(tau-h))) / (2h)`` and the
  Hall-Sheather bandwidth ``h``.  Reproduces Stata ``qreg`` to <=1e-6.
* ``"ker"`` — R ``quantreg::summary.rq(se="ker", hs=TRUE)`` Powell kernel
  sandwich.  Reproduces R to <=1e-6.

See ``methodology/linear/quantile.md`` for the full derivation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as _stats

from open_econs._internal import errors
from open_econs._version import __version__
from open_econs.core.base import BaseModel
from open_econs.core.call_capture import capture_call as _capture_call


def quantile_reg(
    formula: str,
    data: pd.DataFrame,
    tau: float = 0.5,
    method: str = "qreg",
    reps: int = 20,
    seed: int | None = None,
    se_method: str = "stata",
    cov_type: str = "nonrobust",
) -> "QuantileResult":
    """Estimate a quantile (including median) regression.

    Wraps an exact linear-programming solve of the Koenker & Bassett (1978)
    check-function objective — the same solution the Barrodale-Roberts (1974)
    simplex used by Stata ``qreg`` and R ``quantreg::rq(method="br")`` returns.
    Coefficients therefore match both references to machine precision.  Standard
    errors reproduce Stata's default i.i.d. sparsity sandwich or R's Powell
    kernel sandwich, selected by *se_method* (rule-15 toggle), or a paired
    bootstrap for the ``sqreg`` / ``bsqreg`` methods.

    Parameters
    ----------
    formula : str
        Two-sided formula, e.g. ``"y ~ x1 + x2"``.
    data : pd.DataFrame
        Data containing all variables referenced in *formula*.
    tau : float, default 0.5
        Quantile to estimate, strictly between 0 and 1.  The default 0.5 is the
        median (Stata ``qreg`` default).
    method : {"qreg", "sqreg", "bsqreg"}, default "qreg"
        Estimation/VCE flavour, mirroring Stata's family:

        - ``"qreg"``  — analytic sparsity sandwich VCE (see *se_method*).
        - ``"bsqreg"`` — single-quantile paired bootstrap VCE (Stata
          ``bsqreg``); *reps* replications, *seed* for reproducibility.
        - ``"sqreg"`` — simultaneous-quantile bootstrap VCE at a single *tau*
          here (Stata ``sqreg`` with one quantile is ``bsqreg``); the
          between-quantile blocks of full ``sqreg`` are out of scope, so a
          single-*tau* ``sqreg`` is numerically equivalent to ``bsqreg``.
    reps : int, default 20
        Bootstrap replications for ``sqreg`` / ``bsqreg`` (Stata default 20).
    seed : int, optional
        RNG seed for the bootstrap.  REQUIRED for reproducible bootstrap SEs;
        Stata's bootstrap RNG is not portable, so bootstrap SEs are compared
        only to a documented tolerance in the parity tests.
    se_method : {"stata", "ker"}, default "stata"
        Analytic-VCE convention for ``method="qreg"`` (rule-15 toggle):

        - ``"stata"`` (default) — Stata ``qreg`` default VCE
          (``vce(iid, fitted hsheather)``): the Koenker-Bassett sparsity
          sandwich ``V = s^2 * tau*(1-tau) * (X'X)^{-1}`` with the *fitted*
          sparsity ``s = mean(X (b(tau+h) - b(tau-h))) / (2h)`` and the
          Hall-Sheather bandwidth ``h``.  Reproduces Stata to <=1e-6.
        - ``"ker"`` — R ``quantreg::summary.rq(se="ker", hs=TRUE)`` Powell
          Gaussian-kernel sandwich
          ``V = tau*(1-tau) * (X'FX)^{-1} X'X (X'FX)^{-1}`` with
          ``F = diag(phi(u_i/h)/h)`` and R's kernel bandwidth rescaling.
          Reproduces R to <=1e-6.

        Ignored for the bootstrap methods.
    cov_type : {"nonrobust"}, default "nonrobust"
        Reserved for API symmetry with other estimators.  Only ``"nonrobust"``
        is currently supported (the sparsity/kernel sandwich and the bootstrap
        already provide the heteroskedasticity-appropriate VCE).

    Returns
    -------
    QuantileResult
        Immutable result with ``.tidy()``, ``.summary()``, ``.predict()``,
        ``.quantile()``, and ``.vcov()``.

    Examples
    --------
    >>> import open_econs as oe
    >>> r = oe.quantile_reg("y ~ x1 + x2", data=df)                 # median, Stata SE
    >>> r_ker = oe.quantile_reg("y ~ x1 + x2", data=df, se_method="ker")
    >>> r_q25 = oe.quantile_reg("y ~ x1 + x2", data=df, tau=0.25)
    >>> r_bs = oe.quantile_reg("y ~ x1 + x2", data=df, method="bsqreg", seed=1)
    """
    call = _capture_call(
        formula=formula, tau=tau, method=method, reps=reps, seed=seed,
        se_method=se_method, cov_type=cov_type,
    )

    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be strictly between 0 and 1, got {tau!r}.")
    if method not in ("qreg", "sqreg", "bsqreg"):
        raise ValueError(
            f"method must be one of {{'qreg', 'sqreg', 'bsqreg'}}, got {method!r}."
        )
    if se_method not in ("stata", "ker"):
        raise ValueError(
            f"se_method must be one of {{'stata', 'ker'}}, got {se_method!r}."
        )
    if cov_type != "nonrobust":
        raise ValueError(
            f"quantile_reg() supports cov_type='nonrobust' only, got {cov_type!r}. "
            "The sparsity/kernel sandwich (se_method=) and the bootstrap "
            "(method='bsqreg'/'sqreg') already provide the appropriate VCE."
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
            bad_col = m.group(1) if m else rhs_formula
            raise errors.missing_column_error(bad_col, data.columns.tolist()) from e
        raise

    XX = model_spec.rhs if hasattr(model_spec, "rhs") else model_spec
    yy = model_spec.lhs if hasattr(model_spec, "lhs") else None
    if yy is None:
        raise ValueError("quantile_reg() requires a two-sided formula 'y ~ x'.")

    original_n = len(data)
    dropped = original_n - len(yy)
    vars_needed = {str(v) for v in formula_obj.required_variables}
    cols_with_nas = sorted(
        v for v in vars_needed if v in data.columns and data[v].isna().any()
    )
    if dropped > 0:
        import warnings as _w
        _w.warn(
            errors.rows_dropped_warning(dropped, original_n, cols_with_nas),
            RuntimeWarning, stacklevel=3,
        )
    if len(yy) == 0:
        raise errors.empty_data_error(original_n, dropped, cols_with_nas)

    stored_spec = model_spec.model_spec.rhs if hasattr(model_spec, "model_spec") else None

    cols = [str(c) for c in XX.columns]
    X = XX.values.astype(float)
    y = yy.values.ravel().astype(float)
    n, k = X.shape

    beta = _rq_fit(y, X, tau)

    if method in ("sqreg", "bsqreg"):
        cov = _bootstrap_cov(y, X, tau, reps=reps, seed=seed)
        cov_label = f"bootstrap({reps})"
    else:
        if se_method == "ker":
            cov = _cov_ker(y, X, tau, beta)
            cov_label = "iid (kernel)"
        else:
            cov = _cov_stata(y, X, tau)
            cov_label = "iid (fitted, Hall-Sheather)"

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    df_resid = max(n - k, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 0, beta / se, np.nan)
    p_values = 2.0 * _stats.t.sf(np.abs(t_stats), df_resid)
    crit = _stats.t.ppf(0.975, df_resid)
    conf_lower = beta - crit * se
    conf_upper = beta + crit * se

    fitted = X @ beta
    resid = y - fitted

    result = QuantileResult(
        formula=formula,
        rhs_formula=rhs_formula,
        nobs=n,
        df_resid=df_resid,
        df_model=k,
        cov_type=cov_label,
        tau=float(tau),
        method=method,
        se_method=se_method,
        reps=reps if method in ("sqreg", "bsqreg") else 0,
        coefficients=pd.Series(beta, index=cols),
        std_errors=pd.Series(se, index=cols),
        t_stats=pd.Series(t_stats, index=cols),
        p_values=pd.Series(p_values, index=cols),
        conf_int=pd.DataFrame(
            {"lower": conf_lower, "upper": conf_upper}, index=cols
        ),
        fitted=pd.Series(fitted, index=XX.index, name="fitted"),
        residuals=pd.Series(resid, index=XX.index, name="residuals"),
        call=call,
        model_spec=stored_spec,
        _cov=pd.DataFrame(cov, index=cols, columns=cols),
    )
    return result


# ── solver ──────────────────────────────────────────────────────────────────

def _rq_fit(y: np.ndarray, X: np.ndarray, tau: float) -> np.ndarray:
    """Exact quantile-regression coefficients via the check-function LP.

    Solves ``min_b sum_i rho_tau(y_i - x_i'b)`` by the standard LP with
    positive/negative coefficient and residual splits.  SciPy's HiGHS returns
    the same basic (vertex) solution the Barrodale-Roberts simplex finds when
    the optimum is unique, so this matches Stata ``qreg`` / R
    ``rq(method="br")`` to machine precision.
    """
    from scipy.optimize import linprog

    n, k = X.shape
    c = np.concatenate(
        [np.zeros(2 * k), tau * np.ones(n), (1.0 - tau) * np.ones(n)]
    )
    A_eq = np.hstack([X, -X, np.eye(n), -np.eye(n)])
    res = linprog(
        c, A_eq=A_eq, b_eq=y,
        bounds=[(0, None)] * (2 * k + 2 * n),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(
            f"quantile_reg(): linear program failed to solve ({res.message})."
        )
    return res.x[:k] - res.x[k:2 * k]


# ── bandwidth ───────────────────────────────────────────────────────────────

def _hall_sheather_bw(tau: float, n: int, alpha: float = 0.05) -> float:
    """Hall-Sheather (1988) bandwidth (Stata / R ``bandwidth.rq(hs=TRUE)``)."""
    x0 = _stats.norm.ppf(tau)
    f0 = _stats.norm.pdf(x0)
    z = _stats.norm.ppf(1.0 - alpha / 2.0)
    return float(
        n ** (-1.0 / 3.0)
        * z ** (2.0 / 3.0)
        * ((1.5 * f0 ** 2) / (2.0 * x0 ** 2 + 1.0)) ** (1.0 / 3.0)
    )


# ── analytic VCE: Stata default (fitted sparsity) ──────────────────────────

def _cov_stata(y: np.ndarray, X: np.ndarray, tau: float) -> np.ndarray:
    """Stata ``qreg`` default VCE (``vce(iid, fitted hsheather)``).

    ``V = s^2 * tau*(1-tau) * (X'X)^{-1}`` where the *fitted* sparsity is
    ``s = mean(X (b(tau+h) - b(tau-h))) / (2h)`` with the Hall-Sheather
    bandwidth ``h``.  Reproduces Stata ``qreg``'s ``e(V)`` to <=1e-6.
    """
    n, k = X.shape
    h = _hall_sheather_bw(tau, n)
    while (tau - h < 0.0) or (tau + h > 1.0):
        h = h / 2.0
    b_hi = _rq_fit(y, X, tau + h)
    b_lo = _rq_fit(y, X, tau - h)
    dyhat = X @ (b_hi - b_lo)
    sparsity = float(np.mean(dyhat) / (2.0 * h))
    XtX_inv = np.linalg.inv(X.T @ X)
    return sparsity ** 2 * tau * (1.0 - tau) * XtX_inv


# ── analytic VCE: R quantreg kernel (Powell) ───────────────────────────────

def _cov_ker(y: np.ndarray, X: np.ndarray, tau: float, beta: np.ndarray) -> np.ndarray:
    """R ``quantreg::summary.rq(se="ker", hs=TRUE)`` Powell kernel sandwich.

    ``V = tau*(1-tau) * (X'FX)^{-1} X'X (X'FX)^{-1}`` with
    ``F = diag(phi(u_i / h) / h)`` and the kernel bandwidth
    ``h = (Phi^{-1}(tau+h0) - Phi^{-1}(tau-h0)) * min(sd(u), IQR(u)/1.34)``,
    ``h0`` the Hall-Sheather bandwidth.  Reproduces R to <=1e-6.
    """
    n, k = X.shape
    h0 = _hall_sheather_bw(tau, n)
    while (tau - h0 < 0.0) or (tau + h0 > 1.0):
        h0 = h0 / 2.0
    uhat = y - X @ beta
    q1, q3 = np.quantile(uhat, 0.25), np.quantile(uhat, 0.75)
    scale = min(float(np.sqrt(np.var(uhat, ddof=1))), (q3 - q1) / 1.34)
    h = (_stats.norm.ppf(tau + h0) - _stats.norm.ppf(tau - h0)) * scale
    f = _stats.norm.pdf(uhat / h) / h
    fxx = X.T @ (f[:, None] * X)
    fxx_inv = np.linalg.inv(fxx)
    return tau * (1.0 - tau) * fxx_inv @ (X.T @ X) @ fxx_inv


# ── bootstrap VCE (bsqreg / sqreg) ─────────────────────────────────────────

def _bootstrap_cov(
    y: np.ndarray, X: np.ndarray, tau: float, reps: int, seed: int | None,
) -> np.ndarray:
    """Paired (xy) bootstrap VCE for ``bsqreg`` / single-``tau`` ``sqreg``.

    Resamples ``(y_i, x_i)`` pairs with replacement ``reps`` times, refits the
    QR each time, and returns the empirical covariance of the coefficient
    draws.  Matches the *structure* of Stata ``bsqreg``/``sqreg``; the RNG is
    not portable, so exact bootstrap-SE parity is not a target (see
    ``methodology/linear/quantile.md``).
    """
    n, k = X.shape
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, k), dtype=float)
    for b in range(reps):
        idx = rng.integers(0, n, size=n)
        draws[b] = _rq_fit(y[idx], X[idx], tau)
    return np.cov(draws, rowvar=False, ddof=1).reshape(k, k)


class QuantileResult(BaseModel):
    """Result of a quantile (including median) regression.

    Immutable result with the uniform interface (``.tidy()``, ``.summary()``,
    ``.export()``, ``.vcov()``, ``.to_latex()`` / ``.to_html()``). Coefficients
    reproduce Stata ``qreg`` and R ``quantreg::rq(method="br")`` to machine
    precision; standard errors follow the *se_method* convention (Stata default
    sparsity sandwich, R Powell kernel sandwich) or a paired bootstrap for the
    ``bsqreg`` / ``sqreg`` methods.

    Adds ``.quantile()`` (the estimated *tau*) and ``.predict()`` (fitted
    conditional quantile ``x'b``). Inference uses the Student-t distribution
    with ``n - k`` residual degrees of freedom, matching Stata's ``qreg``
    ``t``/``P>|t|`` columns.
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
        tau: float,
        method: str,
        se_method: str,
        reps: int,
        coefficients: pd.Series,
        std_errors: pd.Series,
        t_stats: pd.Series,
        p_values: pd.Series,
        conf_int: pd.DataFrame,
        fitted: pd.Series,
        residuals: pd.Series,
        call: dict[str, Any],
        model_spec: Any = None,
        _cov: pd.DataFrame | None = None,
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
        self.tau = tau
        self.method = method
        self.se_method = se_method
        self.reps = reps
        self.coefficients = coefficients
        self.std_errors = std_errors
        self.t_stats = t_stats
        self.p_values = p_values
        self.conf_int = conf_int
        self.fitted_values = fitted if fitted is not None else pd.Series(dtype=float)
        self.residuals = residuals
        self.model_type = "quantile"
        self._model_spec = model_spec
        self._cov = _cov

        self._freeze()

    def quantile(self) -> float:
        """The estimated quantile *tau*."""
        return self.tau

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
        method_label = {
            "qreg": "Quantile Regression",
            "bsqreg": "Bootstrap Quantile Regression",
            "sqreg": "Simultaneous-Quantile Regression",
        }.get(self.method, "Quantile Regression")
        header = (
            f"                 {method_label} Results                 \n"
            f"======================================================================\n"
            f"Dep. Variable:               {self.formula.split('~')[0].strip()}\n"
            f"Model:                       {method_label}\n"
            f"Quantile (tau):              {self.tau:.4f}\n"
            f"No. Observations:            {self.nobs}\n"
            f"Df Residuals:                {self.df_resid}\n"
            f"Df Model:                    {self.df_model}\n"
            f"Std. Errors:                 {self.cov_type}\n"
            f"======================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return (
            header + tbl +
            "\n======================================================================\n"
        )

    def vcov(self) -> pd.DataFrame:
        if self._cov is None:
            raise RuntimeError("vcov() unavailable: no covariance matrix stored.")
        return self._cov

    def predict(self, newdata: pd.DataFrame | None = None) -> pd.Series:
        """Fitted conditional quantile ``x'b``.

        With ``newdata=None`` returns the in-sample fitted quantile. With
        *newdata* supplied, builds the design matrix from the stored formula
        and returns ``X_new @ beta``.
        """
        if newdata is None:
            return self.fitted_values
        try:
            if self._model_spec is not None:
                matrices = self._model_spec.get_model_matrix(newdata, na_action="drop")
                XX = matrices.rhs if hasattr(matrices, "rhs") else matrices
            else:
                from formulaic import Formula
                matrices = Formula(self.rhs_formula).get_model_matrix(
                    newdata, na_action="drop"
                )
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
        return pd.Series(
            np.dot(XX.values, self.coefficients.values),
            index=XX.index,
            name="predicted",
        )
