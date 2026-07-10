"""Regression-discontinuity design (sharp and fuzzy).

Two code paths:
  1. **rdrobust backend** (default): delegates to the rdrobust Python package
     for CCT bandwidth selection, separate-side local linear estimation,
     and NN cluster-robust variance.  Requires rdrobust >= 2.0.
   2. **Built-in fallback** (``bandwidth_select='ik'``, ``vce='nn'`` or
      ``'ehw'``): corrected Imbens–Kalyanaraman bandwidth, separate-side
      local linear regressions with nearest-neighbour cluster-robust or
      Eicker–Huber–White variance.  No extra dependencies — NN variance is
      implemented natively so parity is not contingent on ``rdrobust``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from open_econs._version import __version__
from open_econs.core.base import BaseModel
from open_econs.core.call_capture import capture_call as _capture_call

try:
    from rdrobust import rdrobust as _rdrobust, rdbwselect as _rdbwselect

    _RDROBUST = True
except ImportError:
    _RDROBUST = False


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


# ── helpers ──────────────────────────────────────────────────────────────────

def _triangular_kernel(x: np.ndarray, c: float, h: float) -> np.ndarray:
    u = np.abs(x - c) / h
    return (1.0 - u) * (u <= 1.0)


def _wls_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares – returns (beta, resid)."""
    XtWX = X.T @ (w[:, None] * X)
    beta = np.linalg.solve(XtWX, X.T @ (w * y))
    resid = y - X @ beta
    return beta, resid


def _ehw_vcov(X: np.ndarray, w: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """Eicker–Huber–White sandwich variance–covariance."""
    scores = X * resid[:, None]
    meat = scores.T @ (w[:, None] ** 2 * scores)
    XtWX = X.T @ (w[:, None] * X)
    return np.linalg.inv(XtWX) @ meat @ np.linalg.inv(XtWX)


def _nn_adjusted_residuals(x: np.ndarray, y: np.ndarray,
                           matches: int = 3) -> np.ndarray:
    """Nearest-neighbour adjusted residuals for NN variance.

    For each observation *i*, ``matches`` nearest neighbours (in the running
    variable) are found — excluding *i* itself.  The adjusted residual is::

        r_i = √(J_i / (J_i + 1)) · (y_i − ȳ_{-i})

    where ``J_i`` is the number of neighbours and ``ȳ_{-i}`` is their
    leave-one-out mean.  This matches ``rdrobust``'s ``vce='nn'``
    convention (``_nn_residuals_jit`` in ``rdrobust.funs``).
    """
    n = len(x)
    idx = np.argsort(x, kind="stable")
    x_s = x[idx]
    y_s = y[idx]
    res = np.zeros(n)
    cap = min(matches, n - 1)
    for pos in range(n):
        left = 0
        right = 0
        while left + right < cap:
            if pos - left - 1 < 0:
                right += 1
            elif pos + right + 1 >= n:
                left += 1
            elif x_s[pos] - x_s[pos - left - 1] > x_s[pos + right + 1] - x_s[pos]:
                right += 1
            elif x_s[pos] - x_s[pos - left - 1] < x_s[pos + right + 1] - x_s[pos]:
                left += 1
            else:
                right += 1
                left += 1
        lo = pos - left
        hi = pos + right + 1
        Ji = (hi - lo) - 1
        if Ji == 0:
            res[pos] = 0.0
        else:
            sf = (Ji / (Ji + 1)) ** 0.5
            y_mean = (np.sum(y_s[lo:hi]) - y_s[pos]) / Ji
            res[pos] = sf * (y_s[pos] - y_mean)
    # Unsort to original order
    res_orig = np.empty(n)
    res_orig[idx] = res
    return res_orig


def _nn_vcov(X: np.ndarray, w: np.ndarray, x_running: np.ndarray,
             y: np.ndarray) -> np.ndarray:
    """Nearest-neighbour sandwich variance (``vce='nn'``).

    Uses NN-adjusted residuals (``_nn_adjusted_residuals``) in place of
    standard regression residuals inside a ``Xᵀ W Σ W X`` sandwich,
    where ``Σ = diag(r²)``.  This matches Stata's ``rdrobust`` with
    ``vce(nn)``.
    """
    resid = _nn_adjusted_residuals(x_running, y)
    scores = X * resid[:, None]
    meat = scores.T @ (w[:, None] ** 2 * scores)
    XtWX = X.T @ (w[:, None] * X)
    return np.linalg.inv(XtWX) @ meat @ np.linalg.inv(XtWX)


# ── built-in bandwidth selectors ─────────────────────────────────────────────

def _ik_bandwidth(x: np.ndarray, y: np.ndarray, c: float) -> float:
    """Imbens–Kalyanaraman (2012) MSE-optimal bandwidth (corrected formula)."""
    n = len(x)
    if n < 10:
        return 1.06 * np.std(x, ddof=1) * n ** (-1.0 / 5.0)

    d = x - c

    # Conditional variance Var(y|x=c)
    try:
        X_var = np.column_stack([np.ones(n), d])
        w_var = _triangular_kernel(x, c, 1.0)
        XtWX = X_var.T @ (w_var[:, None] * X_var)
        be = np.linalg.solve(XtWX, X_var.T @ (w_var * y))
        be2 = np.linalg.solve(XtWX, X_var.T @ (w_var * y ** 2))
        sigma2 = max(float(be2[0]) - float(be[0]) ** 2, 1e-10)
    except Exception:
        sigma2 = float(np.var(y))

    # Density f̂(c)
    try:
        h_s = 1.06 * np.std(x, ddof=1) * n ** (-1.0 / 5.0)
        f_hat = np.sum(_triangular_kernel(x, c, h_s)) / (n * h_s)
        f_hat = max(f_hat, 1e-10)
    except Exception:
        f_hat = 1.0 / (np.max(x) - np.min(x) + 1e-10)

    # Second derivatives from global cubic per side
    try:
        X_poly = np.column_stack([np.ones(n), d, d ** 2, d ** 3])
        left = d < 0
        right = d >= 0
        if np.sum(left) > 4 and np.sum(right) > 4:
            cl = np.linalg.lstsq(X_poly[left], y[left], rcond=None)[0]
            cr = np.linalg.lstsq(X_poly[right], y[right], rcond=None)[0]
            m2_plus, m2_minus = 2.0 * cr[2], 2.0 * cl[2]
        else:
            m2_plus = m2_minus = 0.0
    except Exception:
        m2_plus = m2_minus = 0.0

    C_IK = 3.4375
    m2_sq = m2_plus ** 2 + m2_minus ** 2
    try:
        h_IK = C_IK * (sigma2 / (f_hat * max(m2_sq, 1e-10))) ** (1.0 / 5.0) * n ** (-1.0 / 5.0)
        h_max = (np.max(x) - np.min(x)) / 2.0
        h_IK = np.clip(h_IK, h_max * 0.01, h_max)
    except Exception:
        h_IK = 1.06 * np.std(x, ddof=1) * n ** (-1.0 / 5.0)
    return float(h_IK)


# ── built-in separate-side estimation (sharp & fuzzy) ────────────────────────

def _separate_side_estimates(
    xs: np.ndarray, ys: np.ndarray, d: np.ndarray, w: np.ndarray,
    vce: str = "nn",
) -> tuple[float, float, float, float, int, int]:
    """Separate local linear regressions on each side (sharp RDD).

    Returns (effect, se, z_stat, p_value, n_left, n_right).

    When ``vce='nn'``, the nearest-neighbour cluster-robust variance is
    computed on the *pooled* (4-parameter) model — observations from both
    sides are sorted together so that clusters near the cutoff may contain
    observations from either side, matching Stata's ``rdrobust`` convention.
    """
    left, right = d < 0, d >= 0
    n_l, n_r = int(np.sum(left)), int(np.sum(right))

    if n_l < 2 or n_r < 2:
        return np.nan, np.nan, np.nan, np.nan, n_l, n_r

    X_l = np.column_stack([np.ones(n_l), d[left]])
    beta_l, _ = _wls_fit(X_l, ys[left], w[left])
    a_left = float(beta_l[0])

    X_r = np.column_stack([np.ones(n_r), d[right]])
    beta_r, _ = _wls_fit(X_r, ys[right], w[right])
    a_right = float(beta_r[0])

    effect = a_right - a_left

    if vce == "nn":
        # Per-side NN variance, matching Stata's rdrobust vce(nn):
        # NN-adjusted residuals and sandwich are computed independently
        # on each side.
        V_l = _nn_vcov(X_l, w[left], xs[left], ys[left])
        V_r = _nn_vcov(X_r, w[right], xs[right], ys[right])
        se = float(np.sqrt(max(V_l[0, 0] + V_r[0, 0], 0.0)))
    else:
        beta_l, resid_l = _wls_fit(X_l, ys[left], w[left])
        beta_r, resid_r = _wls_fit(X_r, ys[right], w[right])
        V_l = _ehw_vcov(X_l, w[left], resid_l)
        V_r = _ehw_vcov(X_r, w[right], resid_r)
        se = float(np.sqrt(max(V_l[0, 0] + V_r[0, 0], 0.0)))

    z_stat = effect / se if se > 0 else float("nan")
    p_value = 2.0 * (1.0 - _norm.cdf(abs(z_stat))) if se > 0 else float("nan")
    return effect, se, z_stat, p_value, n_l, n_r


def _fuzzy_ratio_estimates(
    xs: np.ndarray, ys: np.ndarray, tr: np.ndarray,
    d: np.ndarray, w: np.ndarray, vce: str = "nn",
) -> tuple[float, float, float, float, int, int]:
    """Fuzzy RDD via ratio of two sharp estimates (reduced-form / first-stage).

    Returns (effect, se, z_stat, p_value, n_left, n_right).
    """
    n_l, n_r = int(np.sum(d < 0)), int(np.sum(d >= 0))

    efe_fs, se_fs, _, _, _, _ = _separate_side_estimates(xs, tr, d, w, vce=vce)
    efe_rf, se_rf, _, _, _, _ = _separate_side_estimates(xs, ys, d, w, vce=vce)

    if abs(efe_fs) < 1e-12:
        return np.nan, np.nan, np.nan, np.nan, n_l, n_r

    effect = efe_rf / efe_fs
    se = float(np.sqrt(
        se_rf ** 2 / efe_fs ** 2 + efe_rf ** 2 * se_fs ** 2 / efe_fs ** 4
    )) if se_fs > 0 else se_rf / abs(efe_fs)
    z_stat = effect / se if se > 0 else float("nan")
    p_value = 2.0 * (1.0 - _norm.cdf(abs(z_stat))) if se > 0 else float("nan")
    return effect, se, z_stat, p_value, n_l, n_r


# ── public API ───────────────────────────────────────────────────────────────

def rdd(
    data: pd.DataFrame,
    y: str,
    running: str,
    cutoff: float,
    treatment: str | None = None,
    fuzzy: bool = False,
    bandwidth: float | None = None,
    bandwidth_select: str = "cct",
    kernel: str = "triangular",
    vce: str = "nn",
) -> RDResult:
    """Sharp or fuzzy regression-discontinuity design.

    Estimates the discontinuity at ``cutoff`` of the running variable using
    local linear regression with a triangular kernel and, by default, the
    Calonico–Cattaneo–Titiunik MSE-optimal bandwidth and NN cluster-robust
    variance (via the ``rdrobust`` package).

    Parameters
    ----------
    data : pd.DataFrame
        Analysis data.
    y : str
        Outcome column.
    running : str
        Running (forcing) variable column.
    cutoff : float
        Discontinuity threshold.
    treatment : str, optional
        Treatment column.  Required for *fuzzy* RDD.
    fuzzy : bool, default False
        If True, treat the running-variable jump as an instrument for the
        treatment (local-linear-IV fuzzy RDD).
    bandwidth : float, optional
        Half-window width.  If ``None``, selected by ``bandwidth_select``.
    bandwidth_select : {"cct", "ik"}, default "cct"
        ``"cct"`` – Calonico–Cattaneo–Titiunik MSE-optimal (requires
        ``rdrobust``); ``"ik"`` – corrected Imbens–Kalyanaraman.
    kernel : {"triangular"}, default "triangular"
        Only triangular is supported.
    vce : {"nn", "ehw"}, default "nn"
        ``"nn"`` – nearest-neighbour cluster-robust;
        ``"ehw"`` – Eicker–Huber–White.
        ``"nn"`` is always available, regardless of whether ``rdrobust`` is
        installed; the built-in path implements it independently.

    Returns
    -------
    RDResult
    """
    call = _capture_call(
        y=y, running=running, cutoff=cutoff, treatment=treatment,
        fuzzy=fuzzy, bandwidth=bandwidth, bandwidth_select=bandwidth_select,
        kernel=kernel, vce=vce,
    )
    for col in [y, running] + ([treatment] if treatment else []):
        if col not in data.columns:
            from open_econs._internal import errors
            raise errors.missing_column_error(col, data.columns.tolist())
    if fuzzy and treatment is None:
        raise ValueError("Fuzzy RDD requires a `treatment` column.")
    if kernel != "triangular":
        raise ValueError("Only the triangular kernel is supported.")

    x = data[running].values.astype(float)
    y_vals = data[y].values.astype(float)
    tr_vals = data[treatment].values.astype(float) if treatment else None

    # ── rdrobust backend (CCT bandwidth only) ───────────────────────────────
    # Once CCT is computed by rdrobust, the estimate is returned directly
    # (same results as separate-side + NN).  When rdrobust is absent we
    # fall back to the built-in path which now supports both vce="ehw" and
    # vce="nn" independently.
    _VCE_MAP = {"ehw": "hc0", "nn": "nn"}
    use_rd = _RDROBUST and bandwidth_select == "cct"
    if use_rd:
        rd_vce = _VCE_MAP.get(vce, vce)
        rd = _rdrobust(
            y_vals, x, c=cutoff,
            fuzzy=tr_vals if fuzzy else None,
            h=bandwidth,
            kernel="tri",
            bwselect="" if bandwidth is not None else "mserd",
            vce=rd_vce,
        )
        h = float(rd.bws.iloc[0, 0]) if bandwidth is None else bandwidth
        effect = float(rd.Estimate.iloc[0, 0])
        se = float(rd.se.iloc[0, 0])
        z_stat = effect / se if se > 0 else float("nan")
        p_value = 2.0 * (1.0 - _norm.cdf(abs(z_stat))) if se > 0 else float("nan")
        n_left, n_right = int(rd.N_h[0]), int(rd.N_h[1])

    # ── built-in fallback (IK + NN or EHW) ──────────────────────────────────
    else:
        if bandwidth is not None:
            h = bandwidth
        elif bandwidth_select == "cct":
            # CCT requested but rdrobust not installed – fall back to IK
            h = _ik_bandwidth(x, y_vals, cutoff)
        else:
            h = _ik_bandwidth(x, y_vals, cutoff)

        mask = np.abs(x - cutoff) <= h
        xs = x[mask]
        ys = y_vals[mask]
        w = _triangular_kernel(xs, cutoff, h)
        d = xs - cutoff

        if fuzzy:
            trs = tr_vals[mask]
            effect, se, z_stat, p_value, n_left, n_right = _fuzzy_ratio_estimates(
                xs, ys, trs, d, w, vce=vce,
            )
        else:
            effect, se, z_stat, p_value, n_left, n_right = _separate_side_estimates(
                xs, ys, d, w, vce=vce,
            )

    return RDResult(
        running=running, outcome=y, cutoff=cutoff, treatment=treatment,
        fuzzy=fuzzy, bandwidth=float(h), effect=effect, se=se,
        z_stat=z_stat, p_value=p_value, n_left=n_left, n_right=n_right,
        call=call,
    )
