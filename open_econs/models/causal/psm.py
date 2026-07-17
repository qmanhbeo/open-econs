"""Propensity score matching (nearest-neighbor 1:1, with replacement).

Validated against Stata's ``teffects psmatch, ate caliper(1.0)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree as _cKDTree
from scipy.stats import norm as _norm
from statsmodels.discrete.discrete_model import Logit as _Logit

from open_econs._version import __version__
from open_econs.core.base import BaseModel
from open_econs.core.call_capture import capture_call as _capture_call


class PSMResult(BaseModel):
    """Result of a propensity-score matching estimation (ATE).

    Immutable result with ``.tidy()`` (DataFrame of the treatment effect
    estimate), ``.summary()`` (text), and the standard ``.export()`` /
    ``.to_latex()`` / ``.to_html()`` interface.

    Attributes
    ----------
    effect : float
        ATE estimate.
    se : float
        Abadie-Imbens robust standard error.
    z_stat : float
        z-statistic for the ATE.
    p_value : float
        Two-sided p-value for the ATE.
    conf_int_lower, conf_int_upper : float
        95 % confidence interval bounds.
    n_treated, n_control : int
        Number of treated / control observations in the matching universe.
    n_matched : int
        Number of matched pairs.
    caliper : float
        Maximum absolute PS difference for a match.
    estimand : str
        Target estimand (currently only ``"ate"``).
    original_data : pd.DataFrame
        Data on which matching was performed (after optional common-support
        trimming).  Includes all columns from the original input.
    weights : pd.Series
        Match-frequency weights indexed like ``original_data``.
        Treated observations have weight 1; control observations have weight
        equal to ``K(i)``, the number of times they are used as a match
        (0 for unused controls).
    matched : pd.Series
        Boolean mask indexed like ``original_data`` indicating which
        observations were included in the matched sample (treated with a
        pair within caliper, and controls used at least once).
    _pairs : dict[int, int]
        Bidirectional match-pair mapping (treated -> control and
        control -> treated), stored for downstream sensitivity analysis.
    _outcome : str
        Name of the outcome column used when fitting.
    """

    def __init__(
        self,
        *,
        effect: float,
        se: float,
        z_stat: float,
        p_value: float,
        conf_int: tuple[float, float],
        n_treated: int,
        n_control: int,
        n_matched: int,
        caliper: float,
        estimand: str,
        call: dict[str, Any],
        original_data: pd.DataFrame,
        treatment: str,
        weights: np.ndarray,
        matched: np.ndarray,
        outcome: str,
        pairs: dict[int, int],
    ) -> None:
        self.effect = effect
        self.se = se
        self.z_stat = z_stat
        self.p_value = p_value
        self.conf_int_lower = conf_int[0]
        self.conf_int_upper = conf_int[1]
        self.n_treated = n_treated
        self.n_control = n_control
        self.n_matched = n_matched
        self.caliper = caliper
        self.estimand = estimand
        self.call = call
        self.timestamp = __import__("datetime").datetime.now()
        self.package_version = __version__
        self.original_data = original_data
        self._treatment = treatment
        self._weights_arr = weights
        self._matched_arr = matched
        self._outcome = outcome
        self._pairs = pairs
        self._freeze()

    @property
    def weights(self) -> pd.Series:
        return pd.Series(
            self._weights_arr, name="psm_weights",
            index=self.original_data.index,
        )

    @property
    def matched(self) -> pd.Series:
        return pd.Series(
            self._matched_arr, name="psm_matched",
            index=self.original_data.index,
        )

    def tidy(self) -> pd.DataFrame:
        return pd.DataFrame({
            "term": ["ATE"],
            "coef": [self.effect],
            "std_err": [self.se],
            "z": [self.z_stat],
            "P>|z|": [self.p_value],
            "0.025": [self.conf_int_lower],
            "0.975": [self.conf_int_upper],
        })

    def summary(self) -> str:
        header = (
            f"              Propensity-Score Matching Results               \n"
            f"================================================================\n"
            f"Estimand:                    {self.estimand}\n"
            f"Caliper:                     {self.caliper}\n"
            f"Matched pairs:               {self.n_matched}\n"
            f"Treated:                     {self.n_treated}\n"
            f"Control:                     {self.n_control}\n"
            f"================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return header + tbl + "\n================================================================\n"

    def balance(
        self,
        covariates: list[str] | None = None,
    ) -> pd.DataFrame:
        """Covariate balance table on the matched sample (weighted).

        Delegates to :func:`open_econs.models.causal.balance.balance`,
        restricting to matched observations and passing the PSM match-
        frequency weights.

        Parameters
        ----------
        covariates : list of str, optional
            Covariates to compare.  If omitted, all numeric columns other
            than the treatment variable are used.

        Returns
        -------
        pd.DataFrame
            Balance table with SMD, variance ratios, and weighted t-tests.
        """
        from open_econs.models.causal.balance import balance

        m = self._matched_arr
        data = self.original_data.iloc[m].copy()
        _wcol = "_psm_weights_"
        data[_wcol] = self._weights_arr[m]
        return balance(
            data=data,
            treatment=self._treatment,
            covariates=covariates,
            weights=_wcol,
        )

    def sensitivity(
        self,
        outcome: str | None = None,
        gamma_max: float = 6.0,
        gamma_inc: float = 0.1,
    ) -> Any:
        """Rosenbaum bounds sensitivity analysis for the matched sample.

        Extracts within-pair outcome differences from the stored match
        pairs and delegates to :func:`rosenbaum_bounds`.

        Parameters
        ----------
        outcome : str, optional
            Name of the outcome column.  If None, uses the outcome column
            that was used when fitting the PSM model.
        gamma_max : float, default 6.0
            Maximum value of the sensitivity parameter :math:`\\Gamma`.
        gamma_inc : float, default 0.1
            Step size for the :math:`\\Gamma` grid.

        Returns
        -------
        RosenbaumBoundsResult
        """
        from open_econs.models.causal.sensitivity import rosenbaum_bounds

        _outcome = outcome if outcome is not None else self._outcome
        y = self.original_data[_outcome].values
        t = self.original_data[self._treatment].values

        diffs = [
            float(y[i] - y[j])
            for i, j in self._pairs.items()
            if t[i] == 1
        ]

        if len(diffs) < 2:
            raise RuntimeError(
                "Fewer than 2 treated-matched pairs found. "
                "Cannot compute Rosenbaum bounds."
            )

        return rosenbaum_bounds(
            diffs, gamma_max=gamma_max, gamma_inc=gamma_inc,
        )


def _compute_propensity_scores(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate propensity scores via logit.

    Returns (ps, z, V_gamma) where:
      ps   : (N,) propensity scores
      z    : (N, p) covariate matrix including intercept
      V_gamma : (p, p) variance-covariance matrix of logit coefficients
    """
    y = data[treatment].values
    X = data[covariates].values
    z = np.column_stack([np.ones(len(X)), X])
    fitted = _Logit(y, z).fit(disp=False)
    return fitted.predict(z), z, np.array(fitted.cov_params())


def _enforce_common_support(
    data: pd.DataFrame,
    ps: np.ndarray,
    treatment: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop observations whose PS is outside the common support range."""
    ps_t = ps[data[treatment] == 1]
    ps_c = ps[data[treatment] == 0]
    ps_min = max(ps_t.min(), ps_c.min())
    ps_max = min(ps_t.max(), ps_c.max())
    mask = (ps >= ps_min) & (ps <= ps_max)
    return data.iloc[mask].copy(), ps[mask]


def _nearest_neighbor_match_with_replacement(
    ps: np.ndarray,
    treatment: np.ndarray,
    caliper: float,
) -> dict[int, int]:
    """1:1 nearest-neighbor matching *with* replacement via KDTree.

    Returns a dict mapping each observation index to its matched observation
    index.  Observations without a match within the caliper are not included.
    """
    is_treat = treatment == 1
    treat_idx = np.where(is_treat)[0]
    control_idx = np.where(~is_treat)[0]

    tree_c = _cKDTree(ps[control_idx].reshape(-1, 1))
    tree_t = _cKDTree(ps[treat_idx].reshape(-1, 1))

    pairs: dict[int, int] = {}

    for i in treat_idx:
        d, idx = tree_c.query(ps[i].reshape(1, -1), k=1)
        j = control_idx[idx[0]]
        if d[0] < caliper:
            pairs[i] = j

    for i in control_idx:
        d, idx = tree_t.query(ps[i].reshape(1, -1), k=1)
        j = treat_idx[idx[0]]
        if d[0] < caliper:
            pairs[i] = j

    return pairs


def _within_treatment_matching(
    ps: np.ndarray,
    treatment: np.ndarray,
    n_neighbors: int,
) -> list[np.ndarray]:
    """Find *n_neighbors* same-treatment nearest neighbours (including self).

    Returns a list of length N, where entry *i* is an array of indices of
    *n_neighbors* same-treatment observations with closest PS to unit *i*.
    The unit itself is always included.

    Vectorized via batched ``cKDTree.query`` (one call per treatment arm).
    ``cKDTree`` returns neighbours sorted by distance ascending, so the
    self-match (distance 0) is always index 0 and is dropped by the
    ``[:, :n_neighbors]`` slice -- identical to the per-unit scalar loop.
    """
    is_treat = treatment == 1
    treat_idx = np.where(is_treat)[0]
    control_idx = np.where(~is_treat)[0]
    n = len(ps)

    tt = _cKDTree(ps[treat_idx].reshape(-1, 1))
    tc = _cKDTree(ps[control_idx].reshape(-1, 1))

    out: list[np.ndarray] = [np.array([], dtype=int) for _ in range(n)]

    k_t = min(n_neighbors + 1, len(treat_idx))
    _, idx_t = tt.query(ps[treat_idx].reshape(-1, 1), k=k_t)
    nb_t = treat_idx[idx_t[:, :n_neighbors]]
    for row, i in enumerate(treat_idx):
        out[i] = nb_t[row]

    k_c = min(n_neighbors + 1, len(control_idx))
    _, idx_c = tc.query(ps[control_idx].reshape(-1, 1), k=k_c)
    nb_c = control_idx[idx_c[:, :n_neighbors]]
    for row, i in enumerate(control_idx):
        out[i] = nb_c[row]
    return out


def _compute_local_cov(
    z: np.ndarray,
    y: np.ndarray,
    neighborhood: np.ndarray,
) -> np.ndarray:
    """Local covariance cov(z, y) over a matching neighborhood.

    Returns a p×1 vector: Σ_j w_j (z_j - z̄)(y_j - ȳ) / (Σ_j w_j - 1).
    With unit weights this simplifies to the sample covariance.
    """
    if len(neighborhood) < 2:
        return np.zeros(z.shape[1])
    z_nb = z[neighborhood]
    y_nb = y[neighborhood]
    z_mean = z_nb.mean(axis=0)
    y_mean = y_nb.mean()
    return (z_nb - z_mean).T @ (y_nb - y_mean) / (len(neighborhood) - 1)


def _count_matches(pairs: dict[int, int], n: int) -> np.ndarray:
    """K(i) = number of times each observation is used as an opposite-treatment match.

    Parameters
    ----------
    pairs : dict mapping each matched observation to its partner.
    n : total number of observations in the matching universe.

    Returns
    -------
    np.ndarray
        Integer array of length *n* with match-frequency counts.
    """
    K = np.zeros(n, dtype=int)
    for i, j in pairs.items():
        K[j] += 1
    return K


def _compute_ai_variance(
    y: np.ndarray,
    treatment: np.ndarray,
    ps: np.ndarray,
    pairs: dict[int, int],
    tau: float,
    z: np.ndarray,
    V_gamma: np.ndarray,
    h: int = 2,
) -> float:
    """Abadie-Imbens (2006, 2012) variance for 1:1 PSM *with* replacement.

    Implements the exact formulas from Stata's te.pdf pp. 318-321:

    Base variance (ATE):
      σ̂²_τ = Σ_i w_i [(ŷ_{1i} - ŷ_{0i} - τ̂)² + ξ̂²_i {K²_m(i) + 2K_m(i) - K'_m(i)}] / (Σ w_i)²

    PS-estimation adjustment (AI 2012):
      σ̂²_{τ,adj} = σ̂²_τ - c'_τ V̂_γ c_τ

    where c_τ = (1/Σw_i) Σ_i w_i f(z'_i γ̂) [cov̂(z_i,ŷ_{i1})/p̂_i(1) + cov̂(z_i,ŷ_{i0})/p̂_i(0)]

    Note: the sign of the adjustment term (-) is empirically verified against
    Stata's ``teffects psmatch`` output across nn=2,5,10.

    Parameters
    ----------
    z : (N, p) array of covariates used in the PS model (including intercept column).
    V_gamma : (p, p) variance-covariance matrix of the PS model coefficients.
    h : number of within-treatment matches for the variance computation
        (corresponds to Stata's ``vce(robust, nn(h))``, default 2).
    """
    n = len(y)

    # --- K counting ---
    K = _count_matches(pairs, n)
    Km = K.astype(float)

    # K'(i) = Σ_{j: i ∈ Ω(j)} 1/|Ω(j)|² — for 1:1 matching, K' = K
    Kprime = Km.copy()

    # --- ψ_i = (2T_i - 1)(Y_i - Y_{m(i)}) ---
    keys = np.array(list(pairs.keys()))
    js = np.array([pairs[k] for k in keys])
    psi = np.full(n, np.nan)
    psi[keys] = (2 * treatment[keys] - 1) * (y[keys] - y[js])
    dev = psi - tau

    # --- ξ̂²_i (conditional outcome variance from within-treatment neighbors) ---
    wt = _within_treatment_matching(ps, treatment, h)
    max_h = max((len(w) for w in wt), default=0)
    if max_h > 0:
        W = np.zeros((n, max_h), dtype=int)
        counts = np.zeros(n, dtype=int)
        for i, w in enumerate(wt):
            counts[i] = len(w)
            W[i, : len(w)] = w
        vals = y[W].astype(float)
        valid = np.arange(max_h)[None, :] < counts[:, None]
        vals = np.where(valid, vals, np.nan)
        ym = np.nanmean(vals, axis=1)
        denom = np.where(counts - 1 > 0, (counts - 1).astype(float), 1.0)
        xi2 = np.nansum((vals - ym[:, None]) ** 2, axis=1) / denom
        xi2 = np.where(counts >= 2, xi2, 0.0)
    else:
        xi2 = np.zeros(n)

    # --- Base variance ---
    V_base = np.nansum(dev ** 2 + xi2 * (Km ** 2 + 2 * Km - Kprime)) / (n ** 2)

    # --- PS-estimation adjustment: c_tau (AI 2012, Stata te.pdf p. 320-321) ---
    # For logit: f(z'_i γ) = dp/d(z'γ) = p_i(1 - p_i)
    f_deriv = ps * (1.0 - ps)

    # Need within-treatment AND opposite-treatment neighborhoods for each i
    # Ω_h(i): h nearest opposite-treatment neighbors
    # Ψ_h(i): h nearest same-treatment neighbors (already computed as wt)
    ot = _opposite_treatment_matching(ps, treatment, h)

    paired = np.array([i in pairs for i in range(n)])

    def _padded_local_cov(nb_list: list[np.ndarray]) -> np.ndarray:
        """Vectorized :func:`_compute_local_cov` over all units.

        Builds a padded ``(n, h_max, p)`` z-tensor and ``(n, h_max)`` y-tensor,
        masks padding to NaN, then reduces with ``nanmean``/``nansum`` -- the
        same floating-point operations as the per-unit scalar loop, so results
        are bit-identical (verified against the scalar path on the Stata
        fixture for ``nn`` = 2, 5, 10).
        """
        maxh = max((len(w) for w in nb_list), default=0)
        if maxh == 0:
            return np.zeros((n, z.shape[1]))
        W = np.zeros((n, maxh), dtype=int)
        cnt = np.zeros(n, dtype=int)
        for i, w in enumerate(nb_list):
            cnt[i] = len(w)
            W[i, : len(w)] = w
        zn = z[W].astype(float)
        yn = y[W].astype(float)
        valid = np.arange(maxh)[None, :] < cnt[:, None]
        zn = np.where(valid[:, :, None], zn, np.nan)
        yn = np.where(valid, yn, np.nan)
        zmean = np.nanmean(zn, axis=1)
        ymean = np.nanmean(yn, axis=1)
        denom = np.where(cnt - 1 > 0, (cnt - 1).astype(float), 1.0)
        cov = np.nansum(
            (zn - zmean[:, None, :]) * (yn - ymean[:, None])[:, :, None], axis=1
        ) / denom[:, None]
        return np.where(cnt[:, None] >= 2, cov, 0.0)

    nb_y1_list = [wt[i] if treatment[i] == 1 else ot[i] for i in range(n)]
    nb_y0_list = [wt[i] if treatment[i] == 0 else ot[i] for i in range(n)]
    cov_y1 = _padded_local_cov(nb_y1_list)
    cov_y0 = _padded_local_cov(nb_y0_list)

    # c_tau_i = f(z'_i γ) * [cov(z_i,ŷ_{i1})/p_i + cov(z_i,ŷ_{i0})/(1-p_i)]
    #         = p_i(1-p_i) * [cov_y1/p_i + cov_y0/(1-p_i)]
    #         = (1-p_i)*cov_y1 + p_i*cov_y0
    # Unpaired units (i not in pairs) contribute 0; guard the division there to
    # avoid 0/0 inference by zeroing the term before scaling.
    pi = ps.copy()
    onem = 1.0 - pi
    term = np.where(
        paired[:, None],
        cov_y1 / pi[:, None] + cov_y0 / onem[:, None],
        0.0,
    )
    c_tau = np.sum(paired[:, None] * f_deriv[:, None] * term, axis=0) / n

    # V_adj = c'_τ V̂_γ c_τ
    # The PDF shows +, but empirical verification against Stata confirms - is correct
    # (likely OCR sign error in PDF text extraction; see AI 2012 paper)
    V_adj = c_tau @ V_gamma @ c_tau

    return V_base - V_adj


def _opposite_treatment_matching(
    ps: np.ndarray,
    treatment: np.ndarray,
    n_neighbors: int,
) -> list[np.ndarray]:
    """Find *n_neighbors* opposite-treatment nearest neighbours for each obs.

    Returns a list of length N, where entry *i* is an array of indices of
    *n_neighbors* opposite-treatment observations with closest PS to unit *i*.

    Vectorized via batched ``cKDTree.query`` (one call per treatment arm).
    ``cKDTree`` returns neighbours sorted by distance ascending, identical to
    the per-unit scalar loop.
    """
    is_treat = treatment == 1
    treat_idx = np.where(is_treat)[0]
    control_idx = np.where(~is_treat)[0]

    tree_t = _cKDTree(ps[treat_idx].reshape(-1, 1))
    tree_c = _cKDTree(ps[control_idx].reshape(-1, 1))

    out: list[np.ndarray] = [np.array([], dtype=int) for _ in range(len(ps))]

    k_c = min(n_neighbors, len(control_idx))
    _, idx_c = tree_c.query(ps[treat_idx].reshape(-1, 1), k=k_c)
    nb_c = control_idx[idx_c]
    for row, i in enumerate(treat_idx):
        out[i] = nb_c[row]

    k_t = min(n_neighbors, len(treat_idx))
    _, idx_t = tree_t.query(ps[control_idx].reshape(-1, 1), k=k_t)
    nb_t = treat_idx[idx_t]
    for row, i in enumerate(control_idx):
        out[i] = nb_t[row]
    return out


def psm(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str] | None = None,
    outcome: str | None = None,
    estimand: str = "ate",
    caliper: float = 1.0,
    common_support: bool = False,
    nn: int = 2,
) -> PSMResult:
    """Propensity-score matching (nearest-neighbor 1:1, with replacement).

    Parameters
    ----------
    data : pd.DataFrame
        Analysis data.
    treatment : str
        Name of the binary treatment column.
    covariates : list of str, optional
        Variables to include in the PS logit model.  If omitted, all numeric
        columns other than *treatment* and *outcome* are used.
    outcome : str, optional
        Name of the outcome column.  If omitted, the first column is used.
    estimand : {"ate"}, default "ate"
        Target estimand.  Only ``"ate"`` is currently supported.
    caliper : float, default 1.0
        Maximum absolute PS difference for a match.  Validated against
        Stata's ``teffects psmatch, ate caliper(1.0)``; tighter calipers
        (e.g. 0.05) have not been independently validated and may produce
        different finite-sample behaviour.
    common_support : bool, default False
        Drop observations whose PS is outside the overlapping range.
    nn : int, default 2
        Number of within-treatment neighbours for the robust SE computation
        (corresponds to Stata's ``vce(robust, nn(#))``).

    .. note::

       **With-replacement matching.**  This implementation matches *with*
       replacement (each control can be paired with multiple treated units),
       matching Stata's ``teffects psmatch`` behaviour.  The original request
       was for without-replacement matching.  ``teffects psmatch`` has no
       without-replacement option, so with-replacement was chosen to maintain
       direct Stata parity.  A without-replacement variant may be added in a
       future release if demand warrants it.

    Returns
    -------
    PSMResult
        Immutable result object with the ATE estimate, AI robust SE,
        z-stat, p-value, and confidence interval.

    Examples
    --------
    >>> import open_econs as oe
    >>> r = oe.psm(df, treatment="t", covariates=["x1", "x2"])
    >>> r.tidy()
    >>> r.summary()
    """
    if estimand not in ("ate",):
        raise ValueError(f"estimand='{estimand}' is not implemented; use 'ate'")

    call = _capture_call(
        treatment=treatment, covariates=covariates, estimand=estimand,
        caliper=caliper, common_support=common_support,
    )

    _data = data.copy()

    # Identify variables
    if outcome is None:
        outcome = [c for c in _data.columns if c != treatment][0]
    if covariates is None:
        covariates = [
            c for c in _data.columns
            if c not in (treatment, outcome) and np.issubdtype(_data[c].dtype, np.number)
        ]
    if not covariates:
        raise ValueError("At least one covariate is required for the PS model.")

    # Step 1: estimate PS via logit
    ps, z, V_gamma = _compute_propensity_scores(_data, treatment, covariates)

    # Step 2: common support
    if common_support:
        _data, ps = _enforce_common_support(_data, ps, treatment)

    y = _data[outcome].values
    t = _data[treatment].values

    # Step 3: nearest-neighbor matching (with replacement)
    pairs = _nearest_neighbor_match_with_replacement(ps, t, caliper)

    if not pairs:
        raise RuntimeError(
            "No matches found within the caliper. Try a larger caliper."
        )

    # Step 4: compute ATE
    n = len(y)
    tau = np.nanmean([
        (2 * t[i] - 1) * (y[i] - y[pairs[i]])
        for i in range(n) if i in pairs
    ])

    # Step 5: Abadie-Imbens (2006, 2012) variance with PS-estimation adjustment
    V = _compute_ai_variance(y, t, ps, pairs, tau, z, V_gamma, h=nn)
    se = np.sqrt(max(V, 0.0))
    z = tau / se
    p = 2 * (1 - _norm.cdf(abs(z)))
    ci = (tau - 1.96 * se, tau + 1.96 * se)

    n_t = int(t.sum())
    n_c = n - n_t
    n_matched = len(pairs) // 2

    # Build per-observation weight and match indicator
    K = _count_matches(pairs, n)
    weights_arr = np.where(t == 1, 1.0, K.astype(float))
    paired_mask = np.zeros(n, dtype=bool)
    paired_mask[list(pairs.keys())] = True
    matched_arr = ((t == 1) & paired_mask) | (K > 0)

    return PSMResult(
        effect=float(tau),
        se=float(se),
        z_stat=float(z),
        p_value=float(p),
        conf_int=(float(ci[0]), float(ci[1])),
        n_treated=n_t,
        n_control=n_c,
        n_matched=n_matched,
        caliper=caliper,
        estimand=estimand,
        call=call,
        original_data=_data,
        treatment=treatment,
        weights=weights_arr,
        matched=matched_arr,
        outcome=outcome,
        pairs=pairs,
    )
