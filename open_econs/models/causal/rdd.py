from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from open_econs._version import __version__
from open_econs.core.base import BaseModel
from open_econs.core.call_capture import capture_call as _capture_call


class RDResult(BaseModel):
    """Result of a regression-discontinuity design (sharp or fuzzy)."""

    def __init__(
        self,
        *,
        running: str,
        outcome: str,
        cutoff: float,
        treatment: str | None,
        fuzzy: bool,
        bandwidth: float,
        effect: float,
        se: float,
        z_stat: float,
        p_value: float,
        n_left: int,
        n_right: int,
        call: dict[str, Any],
    ) -> None:
        self.running = running
        self.outcome = outcome
        self.cutoff = cutoff
        self.treatment = treatment
        self.fuzzy = fuzzy
        self.bandwidth = bandwidth
        self.effect = effect
        self.se = se
        self.z_stat = z_stat
        self.p_value = p_value
        self.n_left = n_left
        self.n_right = n_right
        self.call = call
        self.timestamp = __import__("datetime").datetime.now()
        self.package_version = __version__
        self._freeze()

    def tidy(self) -> pd.DataFrame:
        return pd.DataFrame({
            "term": ["discontinuity"],
            "coef": [self.effect],
            "std_err": [self.se],
            "z": [self.z_stat],
            "P>|z|": [self.p_value],
        })

    def summary(self) -> str:
        kind = "Fuzzy" if self.fuzzy else "Sharp"
        return (
            f"            {kind} Regression Discontinuity Results            \n"
            f"==================================================================\n"
            f"  Running variable : {self.running}\n"
            f"  Outcome          : {self.outcome}\n"
            f"  Cutoff           : {self.cutoff}\n"
            f"  Bandwidth (h)    : {self.bandwidth:.4f}\n"
            f"  Observations     : {self.n_left} left / {self.n_right} right\n"
            f"  Discontinuity    : {self.effect:.4f} (se {self.se:.4f}, "
            f"p {self.p_value:.3f})\n"
        )


def _local_linear(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares.  X is (n, k), returns (coef, vcov)."""
    W = np.diag(w)
    XtW = X.T @ W
    XtWX = XtW @ X
    XtWy = XtW @ y
    coef = np.linalg.solve(XtWX, XtWy)
    resid = y - X @ coef
    # Heteroskedasticity-robust (Eicker-Huber-White) vcov.
    scores = X * resid[:, None]
    S = scores.T @ (w[:, None] * scores)
    V = np.linalg.inv(XtWX) @ S @ np.linalg.inv(XtWX)
    return coef, V


def _triangular_kernel(x: np.ndarray, c: float, h: float) -> np.ndarray:
    u = np.abs(x - c) / h
    w = (1.0 - u) * (u <= 1.0)
    return w


def rdd(
    data: pd.DataFrame,
    y: str,
    running: str,
    cutoff: float,
    treatment: str | None = None,
    fuzzy: bool = False,
    bandwidth: float | None = None,
    kernel: str = "triangular",
) -> RDResult:
    """Sharp or fuzzy regression-discontinuity design via local linear regression.

    Estimates the discontinuity in the outcome at the cutoff of a running
    variable, using a triangular-kernel-weighted local linear regression on
    each side of the cutoff (the Imbens-Kalyanaraman style local polynomial).

    Parameters
    ----------
    data : pd.DataFrame
        Analysis data.
    y : str
        Outcome column.
    running : str
        Running variable column.
    cutoff : float
        Discontinuity threshold.
    treatment : str, optional
        Treatment-indicator column.  Required for *fuzzy* RDD (the treatment
        does not jump exactly at the cutoff); ignored for *sharp* RDD where the
        treatment is defined as ``running >= cutoff``.
    fuzzy : bool, default False
        If True, use the running-variable jump at the cutoff as an instrument
        for ``treatment`` (local linear IV / 2SLS).
    bandwidth : float, optional
        Half-window width around the cutoff.  Defaults to a simple
        rule-of-thumb ``1.06 * std(x) * n**(-1/5)``.
    kernel : {"triangular"}, default "triangular"
        Kernel for local weighting (triangular is standard for RDD).

    Returns
    -------
    RDResult
        Immutable result with the discontinuity estimate, SE, z-stat, p-value,
        and the number of observations on each side of the cutoff.
    """
    call = _capture_call(
        y=y, running=running, cutoff=cutoff, treatment=treatment,
        fuzzy=fuzzy, bandwidth=bandwidth, kernel=kernel,
    )
    for col in ([y, running] + ([treatment] if treatment else [])):
        if col not in data.columns:
            from open_econs._internal import errors

            raise errors.missing_column_error(col, data.columns.tolist())
    if fuzzy and treatment is None:
        raise ValueError("Fuzzy RDD requires a `treatment` column.")

    x = data[running].values.astype(float)
    if bandwidth is None:
        bandwidth = 1.06 * np.std(x, ddof=1) * len(x) ** (-1.0 / 5.0)
    if kernel != "triangular":
        raise ValueError("Only the triangular kernel is supported.")

    mask = np.abs(x - cutoff) <= bandwidth
    xs = x[mask]
    ys = data[y].values.astype(float)[mask]
    w = _triangular_kernel(xs, cutoff, bandwidth)
    d = xs - cutoff  # centered running variable

    if fuzzy:
        tr = data[treatment].values.astype(float)[mask]
        # Local linear IV: y ~ treated + d, instrumented with 1{d >= 0},
        # plus the centered running variable as a covariate on both stages.
        z = (d >= 0).astype(float)
        X_endo = np.column_stack([tr, d])
        Z = np.column_stack([z, d])
        W = np.diag(w)
        ZtWZ = Z.T @ W @ Z
        G = np.linalg.inv(ZtWZ)
        H = Z @ G @ Z.T  # (n, n), (P_Z W)_{ij}
        M = W @ H @ W
        A = X_endo.T @ M @ X_endo
        beta = np.linalg.solve(A, X_endo.T @ M @ ys)
        effect = float(beta[0])
        u = ys - X_endo @ beta
        # Robust IV variance: A^{-1} [Σ w_i u_i^2 (P_Z W X_i)(P_Z W X_i)'] A^{-1}.
        pwx = (H * w[:, None]) @ X_endo  # (n, k_endo) = (P_Z W X)_i rows
        middle = np.zeros((X_endo.shape[1], X_endo.shape[1]))
        for i in range(len(ys)):
            middle += (w[i] * u[i] ** 2) * np.outer(pwx[i], pwx[i])
        V = np.linalg.inv(A) @ middle @ np.linalg.inv(A)
        se = float(np.sqrt(max(V[0, 0], 0.0)))
        z_stat = float(effect / se) if se > 0 else float("nan")
        p_value = float(2 * (1 - _norm.cdf(abs(z_stat)))) if se > 0 else float("nan")
    else:
        treated = (d >= 0).astype(float)
        X = np.column_stack([treated, d])
        beta, V = _local_linear(X, ys, w)
        effect = float(beta[0])
        se = float(np.sqrt(max(V[0, 0], 0.0)))
        z_stat = float(effect / se) if se > 0 else float("nan")
        p_value = float(2 * (1 - _norm.cdf(abs(z_stat)))) if se > 0 else float("nan")

    n_left = int(np.sum(d < 0))
    n_right = int(np.sum(d >= 0))
    return RDResult(
        running=running, outcome=y, cutoff=cutoff, treatment=treatment,
        fuzzy=fuzzy, bandwidth=float(bandwidth), effect=effect, se=se,
        z_stat=z_stat, p_value=p_value, n_left=n_left, n_right=n_right,
        call=call,
    )
