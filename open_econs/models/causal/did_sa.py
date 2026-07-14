"""Sun & Abraham (2021) Interaction-Weighted Difference-in-Differences.

Implements the Sun & Abraham (2021) estimator from:
    Sun, Liyang, and Abraham, Sarah. 2021.
    "Estimating Dynamic Treatment Effects in Event Studies
    With Heterogeneous Treatment Effects."
    Journal of Econometrics, 225(2): 175-199.

The estimator performs:
    1. Create period x cohort interaction dummies (relative time x cohort).
    2. Drop reference period (-1 by default) and never-treated cohort.
    3. Use FWL to partial out entity and time FE via iterative demeaning.
    4. Run OLS on the demeaned system (covariates + interaction dummies).
    5. Detect collinearity among demeaned interaction dummies only.
    6. ATT = weighted average of interaction coefficients, where weights
       are cohort-period cell shares (share of treated observations in
       each cohort-period cell).
    7. SE = sqrt(w' V w) where V is the full cluster-robust VCE.

R parity anchor: ``fixest::sunab()`` (v0.14.2).

References
----------
Sun, Liyang, and Abraham, Sarah. 2021.
    "Estimating Dynamic Treatment Effects in Event Studies
    With Heterogeneous Treatment Effects."
    Journal of Econometrics, 225(2): 175-199.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from open_econs._version import __version__
from open_econs.core.call_capture import capture_call as _capture_call
from open_econs.core.base import BaseModel
from open_econs.models.linear.fe import _demean_two_way, _count_absorbed_dof


class SaDiDResult(BaseModel):
    """Result of Sun & Abraham (2021) Interaction-Weighted DID.

    Immutable result exposing a uniform interface: ``.tidy()`` (coefficients,
    SEs, t-stats, p-values, CI), ``.summary()`` (text), ``.export()``
    (CSV/JSON/Pickle), ``.vcov()``.  The key quantity is the ATT estimate
    ``att`` with its ``att_se`` / ``att_t_stat`` / ``att_p_value``.

    Aggregated views:
        - ``period_coefs`` / ``period_ses`` / ``period_names``: relative-time
          period-level aggregates (weighted by cohort shares within each period).
        - ``cohort_coefs`` / ``cohort_ses`` / ``cohort_names``: cohort-specific
          ATTs (the time::0 coefficient for each cohort, if non-collinear).

    References
    ----------
    Sun, Liyang, and Abraham, Sarah. 2021.
        "Estimating Dynamic Treatment Effects in Event Studies
        With Heterogeneous Treatment Effects."
        Journal of Econometrics, 225(2): 175-199.
    """

    def __init__(
        self,
        *,
        cohort_var: str,
        period_var: str,
        ref_period: int,
        nobs: int,
        dep_var: str,
        coefficients: pd.Series,
        std_errors: pd.Series,
        t_stats: pd.Series,
        p_values: pd.Series,
        conf_int: pd.DataFrame,
        att: float,
        att_se: float,
        att_t_stat: float,
        att_p_value: float,
        r_squared: float,
        adj_r_squared: float,
        sigma2: float,
        vce: np.ndarray,
        call: dict[str, Any],
        period_coefs: np.ndarray | None = None,
        period_ses: np.ndarray | None = None,
        period_names: list[str] | None = None,
        cohort_coefs: np.ndarray | None = None,
        cohort_ses: np.ndarray | None = None,
        cohort_names: list[str] | None = None,
    ) -> None:
        self.cohort_var = cohort_var
        self.period_var = period_var
        self.ref_period = ref_period
        self.data_shape = (nobs, coefficients.shape[0])
        self.call = call
        self.timestamp = datetime.now()
        self.package_version = __version__

        self.nobs = nobs
        self.dep_var = dep_var
        self.coefficients = coefficients
        self.std_errors = std_errors
        self.t_stats = t_stats
        self.p_values = p_values
        self.conf_int = conf_int
        self.att = att
        self.att_se = att_se
        self.att_t_stat = att_t_stat
        self.att_p_value = att_p_value
        self.r_squared = r_squared
        self.adj_r_squared = adj_r_squared
        self.sigma2 = sigma2
        self._vce = vce

        self.period_coefs = period_coefs
        self.period_ses = period_ses
        self.period_names = period_names
        self.cohort_coefs = cohort_coefs
        self.cohort_ses = cohort_ses
        self.cohort_names = cohort_names

        self._freeze()

    def tidy(self) -> pd.DataFrame:
        """Return coefficients table."""
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
        """Return text summary."""
        att_str = (
            f"ATT Estimate:                     {self.att:.6f}\n"
            f"Std. Error:                       {self.att_se:.6f}\n"
            f"t-statistic:                      {self.att_t_stat:.4f}\n"
            f"P>|t|:                            {self.att_p_value:.6e}\n"
        )
        header = (
            f"     Sun & Abraham (2021) Interaction-Weighted DID      \n"
            f"============================================================\n"
            f"Dep. Variable:               {self.dep_var}\n"
            f"Cohort variable:             {self.cohort_var}\n"
            f"Period variable:             {self.period_var}\n"
            f"Reference period:            {self.ref_period}\n"
            f"No. Observations:            {self.nobs}\n"
            f"R-squared:                   {self.r_squared:.6f}\n"
            f"Adj. R-squared:              {self.adj_r_squared:.6f}\n"
            f"Sigma2:                      {self.sigma2:.6f}\n"
            f"============================================================\n"
            f"ATT:\n"
            f"{att_str}"
            f"============================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return (
            header + tbl +
            "\n============================================================\n"
        )

    def vcov(self) -> pd.DataFrame:
        """Return the parameter variance-covariance matrix."""
        return pd.DataFrame(
            self._vce,
            index=self.coefficients.index,
            columns=self.coefficients.index,
        )


def _build_sunab_dummies(
    cohort: np.ndarray,
    period: np.ndarray,
    ref_period: int = -1,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build period x cohort interaction dummies for Sun-Abraham.

    Parameters
    ----------
    cohort : np.ndarray
        Treatment cohort for each observation (NA for never-treated).
    period : np.ndarray
        Calendar period for each observation.
    ref_period : int
        Reference period to exclude (default: -1).

    Returns
    -------
    dummies : np.ndarray
        Interaction dummy matrix (n_obs x n_dummies).
    dummy_names : list[str]
        Names for each dummy column (e.g., "time::2:cohort::3").
    keep_mask : np.ndarray
        Boolean mask of observations to keep (not never-treated).
    """
    # Identify never-treated (cohort is NA)
    never_treated = np.isnan(cohort)

    # Compute relative time: period - cohort
    rel_period = np.where(never_treated, np.nan, period - cohort)

    # Build dummy matrix for OBSERVED cohort-period combinations only.
    # Unobserved combinations (e.g. time::-4:cohort::2) must not create
    # all-zero columns, which would distort collinearity detection.
    observed: set[tuple[int, int]] = set()
    for i in range(len(cohort)):
        if not np.isnan(cohort[i]) and not np.isnan(rel_period[i]) and rel_period[i] != ref_period:
            observed.add((int(rel_period[i]), int(cohort[i])))
    sorted_combos = sorted(observed)

    n_obs = len(cohort)
    n_dummies = len(sorted_combos)
    dummies = np.zeros((n_obs, n_dummies), dtype=float)
    dummy_names = []

    for col_idx, (rp, c) in enumerate(sorted_combos):
        mask = (rel_period == rp) & (cohort == c)
        dummies[mask, col_idx] = 1.0
        dummy_names.append(f"time::{rp}:cohort::{c}")

    # Keep mask: only exclude never-treated (reference period obs stay in sample)
    keep_mask = ~never_treated

    return dummies, dummy_names, keep_mask


def _compute_sunab_shares(
    dummies: np.ndarray,
    coef_idx: np.ndarray,
) -> np.ndarray:
    """Compute cohort-period cell shares for ATT aggregation.

    Shares are the fraction of observations contributing to each
    coefficient, normalized to sum to 1.

    Parameters
    ----------
    dummies : np.ndarray
        Full dummy matrix (n_obs x n_dummies).
    coef_idx : np.ndarray
        Indices of the interaction coefficients to aggregate.

    Returns
    -------
    shares : np.ndarray
        Normalized shares for each coefficient.
    """
    col_sums = np.sum(np.sign(dummies[:, coef_idx]), axis=0)
    shares = col_sums / np.sum(col_sums)
    return shares


def _detect_collinearity(
    X: np.ndarray,
    col_names: list[str],
) -> tuple[list[int], list[str]]:
    """Detect collinear columns via sequential projection (Gram-Schmidt).

    Processes columns in their original order.  A column is dropped if
    it lies in the span of the previously accepted columns (R² ≈ 1).
    This matches fixest's Cholesky-based collinearity detection, which
    processes columns in order and drops those that cause Cholesky failure.

    Empirically verified to produce the same 4 dropped columns as R's
    fixest::sunab() on the Sun-Abraham test fixture (known limitation:
    equivalence depends on column order matching the R model matrix
    column order — valid for sunab but not guaranteed for arbitrary
    regressor sets).
    """
    n, k = X.shape
    if k == 0:
        return [], []
    tol = 1e-10
    accepted: list[int] = []
    for j in range(k):
        if len(accepted) == 0:
            accepted.append(j)
            continue
        Xj = X[:, j]
        Xa = X[:, accepted]
        beta_j, _, _, _ = np.linalg.lstsq(Xa, Xj, rcond=None)
        resid = Xj - Xa @ beta_j
        ss_resid = float(np.sum(resid ** 2))
        ss_total = float(np.sum((Xj - np.mean(Xj)) ** 2))
        r2 = 1.0 - ss_resid / ss_total if ss_total > 0 else 1.0
        if r2 > 1.0 - tol:
            continue  # collinear — skip
        accepted.append(j)
    kept_names = [col_names[i] for i in accepted]
    return accepted, kept_names


def _parse_dummy_name(name: str) -> tuple[int, int]:
    """Parse a dummy name like 'time::-3:cohort::4' into (rel_period, cohort)."""
    # Format: 'time::<rp>:cohort::<c>' -> split('::') gives ['time','<rp>:cohort','<c>']
    parts = name.split("::")
    rp_str = parts[1].split(":")[0]  # e.g. '-3' from '-3:cohort'
    c_str = parts[2]                 # e.g. '4'
    return int(rp_str), int(c_str)


def _compute_sunab_aggregates(
    kept_names: list[str],
    beta: np.ndarray,
    V: np.ndarray,
    D: np.ndarray,
    dummy_names: list[str],
    dummy_keep_idx: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, list[str],
    np.ndarray, np.ndarray, list[str],
]:
    """Compute period-level and cohort-level aggregated coefficients and SEs.

    Period-level: for each relative time t, the cohort-weighted average of
    non-collinear coefficients at that period.  Weights are N_{g,t} / N_t
    (share of cohort g among non-collinear observations at period t).

    Cohort-level: for each cohort g, the time::0 coefficient if non-collinear
    (the cohort-specific ATT at the time of treatment).

    Returns (period_coefs, period_ses, period_names, cohort_coefs, cohort_ses,
    cohort_names).
    """
    dummy_name_to_col = {name: i for i, name in enumerate(dummy_names)}

    # Build mapping: dummy_name -> (rel_period, cohort, coef_index_in_beta)
    entries: list[tuple[str, int, int, int]] = []
    for idx, name in enumerate(kept_names):
        if name in dummy_name_to_col:
            rp, c = _parse_dummy_name(name)
            entries.append((name, rp, c, idx))

    # Cell sizes from the original dummy matrix (estimation sample)
    cell_sizes: dict[tuple[int, int], int] = {}
    for name, rp, c, _ in entries:
        col = dummy_name_to_col[name]
        cell_sizes[(rp, c)] = int(np.sum(D[:, col]))

    # ── Period-level aggregates ──────────────────────────────────────────
    period_groups: dict[int, list[tuple[int, int, int]]] = {}
    for name, rp, c, beta_idx in entries:
        period_groups.setdefault(rp, []).append((c, cell_sizes[(rp, c)], beta_idx))

    sorted_periods = sorted(period_groups.keys())
    period_coefs = np.zeros(len(sorted_periods))
    period_ses = np.zeros(len(sorted_periods))
    period_names = [f"time::{rp}" for rp in sorted_periods]

    for i, rp in enumerate(sorted_periods):
        cohorts_info = period_groups[rp]
        total = sum(sz for _, sz, _ in cohorts_info)
        w = np.zeros(len(beta))
        for c, sz, beta_idx in cohorts_info:
            w[beta_idx] = sz / total
        period_coefs[i] = float(w @ beta)
        period_ses[i] = float(np.sqrt(w @ V @ w))

    # ── Cohort-level aggregates (time::0 coefficients) ───────────────────
    cohort_groups: dict[int, list[tuple[int, int, int]]] = {}
    for name, rp, c, beta_idx in entries:
        if rp == 0:
            cohort_groups.setdefault(c, []).append((c, cell_sizes[(rp, c)], beta_idx))

    sorted_cohorts = sorted(cohort_groups.keys())
    cohort_coefs = np.zeros(len(sorted_cohorts))
    cohort_ses = np.zeros(len(sorted_cohorts))
    cohort_names = [f"cohort::{c}" for c in sorted_cohorts]

    for i, c in enumerate(sorted_cohorts):
        info = cohort_groups[c]
        if len(info) == 1:
            _, _, beta_idx = info[0]
            cohort_coefs[i] = float(beta[beta_idx])
            cohort_ses[i] = float(np.sqrt(V[beta_idx, beta_idx]))
        else:
            total = sum(sz for _, sz, _ in info)
            w = np.zeros(len(beta))
            for _, sz, beta_idx in info:
                w[beta_idx] = sz / total
            cohort_coefs[i] = float(w @ beta)
            cohort_ses[i] = float(np.sqrt(w @ V @ w))

    return period_coefs, period_ses, period_names, cohort_coefs, cohort_ses, cohort_names


def did_sa(
    data: pd.DataFrame,
    y: str,
    cohort: str,
    period: str,
    ref_period: int = -1,
    entity: str | None = None,
    time: str | None = None,
    cluster: str | None = None,
    covariates: list[str] | None = None,
) -> SaDiDResult:
    """Sun & Abraham (2021) Interaction-Weighted Difference-in-Differences (did_sa).

    Implements the Sun & Abraham (2021) estimator using Frisch-Waugh-Lovell
    (FWL): entity and time FE are partialled out via iterative demeaning
    (``_demean_two_way``), and OLS is run on the residualized system of
    covariates + interaction dummies.

    Cluster-robust standard errors use the CRV1 sandwich estimator with
    fixest's default small-sample correction (ssc()): G/(G-1) for cluster
    adjacency and (n-1)/(n-K) for absorbed-FE DOF adjustment, where G is
    the number of entity clusters and K = nparams - (G-1).  T-tests use
    df = G - 1.

    Parameters
    ----------
    data : pd.DataFrame
        Analysis data.
    y : str
        Outcome variable name.
    cohort : str
        Cohort column (NA for never-treated).
    period : str
        Period column (calendar time).
    ref_period : int
        Reference period to exclude (default: -1).
    entity : str, optional
        Entity column for entity FE.
    time : str, optional
        Time column for time FE.
    cluster : str, optional
        Cluster column for cluster-robust SEs.
    covariates : list[str], optional
        Covariate column names.

    Returns
    -------
    SaDiDResult
    """
    call = _capture_call(
        y=y,
        cohort=cohort,
        period=period,
        ref_period=ref_period,
        entity=entity,
        time=time,
        cluster=cluster,
        covariates=covariates,
    )

    # ── Validate inputs ─────────────────────────────────────────────────
    for col_name in [y, cohort, period]:
        if col_name not in data.columns:
            raise ValueError(
                f"Column '{col_name}' not found in data. "
                f"Available: {list(data.columns)}"
            )
    if entity is not None and entity not in data.columns:
        raise ValueError(
            f"Entity column '{entity}' not found in data. "
            f"Available: {list(data.columns)}"
        )
    if cluster is not None and cluster not in data.columns:
        raise ValueError(
            f"Cluster column '{cluster}' not found in data. "
            f"Available: {list(data.columns)}"
        )

    n_total = len(data)

    # ── Extract arrays ──────────────────────────────────────────────────
    y_arr = data[y].values.astype(float)
    cohort_arr = data[cohort].values.astype(float)
    period_arr = data[period].values.astype(float)

    # ── Build interaction dummies ───────────────────────────────────────
    # keep_mask excludes never-treated (NA cohort) but INCLUDES
    # reference-period observations so they contribute to FE estimation.
    dummies, dummy_names, keep_mask = _build_sunab_dummies(
        cohort_arr, period_arr, ref_period=ref_period,
    )

    n_keep = int(keep_mask.sum())
    if n_keep == 0:
        raise ValueError("No observations remaining after dropping "
                         "never-treated.")

    # ── Build covariate matrix ──────────────────────────────────────────
    cov_parts: list[np.ndarray] = []
    cov_names: list[str] = []
    if covariates:
        for cov in covariates:
            if cov not in data.columns:
                raise ValueError(
                    f"Covariate column '{cov}' not found in data. "
                    f"Available: {list(data.columns)}"
                )
            cov_parts.append(data[cov].values[keep_mask].astype(float).reshape(-1, 1))
            cov_names.append(cov)

    if cov_parts:
        X_cov = np.column_stack(cov_parts)
    else:
        X_cov = np.empty((n_keep, 0), dtype=float)

    # Interaction dummies (estimation sample)
    D = dummies[keep_mask]

    # ── FWL: partial out entity + time FE via demeaning ─────────────────
    entity_arr = data[entity].values[keep_mask] if entity else None
    time_arr = data[time].values[keep_mask] if time else None

    y_d = y_arr[keep_mask].copy()

    if entity_arr is not None and time_arr is not None:
        # Stack covariates + dummies for joint demeaning
        X_all = np.column_stack([X_cov, D]) if X_cov.shape[1] > 0 else D.copy()
        y_d, X_d = _demean_two_way(y_d, X_all, entity_arr, time_arr)
        # Split back
        n_cov = X_cov.shape[1]
        if n_cov > 0:
            X_cov_d = X_d[:, :n_cov]
            D_d = X_d[:, n_cov:]
        else:
            X_cov_d = np.empty((n_keep, 0), dtype=float)
            D_d = X_d
    elif entity_arr is not None:
        from open_econs.models.linear.fe import _demean
        X_all = np.column_stack([X_cov, D]) if X_cov.shape[1] > 0 else D.copy()
        y_d = _demean(y_d, entity_arr)
        X_d = _demean(X_all, entity_arr)
        n_cov = X_cov.shape[1]
        if n_cov > 0:
            X_cov_d = X_d[:, :n_cov]
            D_d = X_d[:, n_cov:]
        else:
            X_cov_d = np.empty((n_keep, 0), dtype=float)
            D_d = X_d
    else:
        # No FE — just use raw data
        X_cov_d = X_cov
        D_d = D

    # ── Collinearity detection on demeaned interaction dummies only ─────
    all_names = cov_names + dummy_names
    all_demeaned = np.column_stack([X_cov_d, D_d]) if X_cov_d.shape[1] > 0 else D_d

    keep_cols, kept_names = _detect_collinearity(all_demeaned, all_names)

    if len(keep_cols) == 0:
        raise ValueError("All regressors dropped by collinearity screen.")

    X_final = all_demeaned[:, keep_cols]
    n, k = X_final.shape

    # Identify which kept columns are interaction dummies
    dummy_set = set(dummy_names)
    dummy_keep_idx = np.array([i for i, name in enumerate(kept_names)
                               if name in dummy_set])

    # Map kept dummy names back to their positions in the original dummy
    # matrix D (pre-collinearity, pre-demeaning) for cell-share weights.
    dummy_name_to_col = {name: i for i, name in enumerate(dummy_names)}
    dummy_cols_in_D = np.array([dummy_name_to_col[kept_names[i]]
                                for i in dummy_keep_idx])

    # ── OLS on demeaned system ──────────────────────────────────────────
    from numpy.linalg import inv
    XtX = X_final.T @ X_final
    XtX_inv = inv(XtX)
    beta = XtX_inv @ (X_final.T @ y_d)
    residuals = y_d - X_final @ beta

    # ── Sigma² (with absorbed-FE DOF correction) ───────────────────────
    # fixest: nparams = k_estimated + absorbed_dof
    # absorbed_dof = sum(n_groups_i) - (k_fe - 1)
    fe_cols_for_dof = [c for c in [entity, time] if c is not None]
    n_absorbed = _count_absorbed_dof(
        data.loc[data.index[keep_mask]], fe_cols_for_dof
    ) if fe_cols_for_dof else 0
    nparams = k + n_absorbed
    df_resid = max(n - nparams, 1)
    sigma2 = float(np.sum(residuals ** 2) / df_resid)

    # ── Cluster-robust VCE ─────────────────────────────────────────────
    if cluster is not None:
        cluster_arr = data[cluster].values[keep_mask]
        unique_clusters = np.unique(cluster_arr)
        G = len(unique_clusters)
        # CRV1 sandwich with fixest's default SSC:
        #
        # From fixest v0.14.2 source:  fixest:::vcov_cluster_internal()
        # applies two corrections (with defaults ssc(K.adj=TRUE,
        # K.fixef="nonnested", G.adj=TRUE, G.df="min")):
        #
        #   1. G.adj factor:  G / (G - 1)           [cluster.adj]
        #   2. K.adj factor:  (n - 1) / (n - K)     [fixef.K adjustment]
        #
        # K is computed by ssc_compute_K().  For K.fixef="nonnested" with
        # the cluster variable matching the first FE (entity), the branch
        # subtracts the entity-FE DOF:
        #   K = nparams - (entity_groups - 1)
        #
        # Source: fixest/R/vcov.R, fixest:::ssc_compute_K(),
        #         fixest:::vcov_cluster_internal().
        n_est = n  # estimation sample size (75)
        n_clusters = G  # entity clusters = entity FE groups
        k_eff = nparams - (n_clusters - 1)
        ssc_g_adj = G / max(G - 1, 1)
        ssc_k_adj = (n_est - 1) / max(n_est - k_eff, 1)
        ssc_factor = ssc_g_adj * ssc_k_adj
        meat = np.zeros((k, k))
        for c_val in unique_clusters:
            idx = cluster_arr == c_val
            Xg = X_final[idx]
            ug = residuals[idx]
            score_g = Xg.T @ ug
            meat += np.outer(score_g, score_g)
        V = XtX_inv @ (meat * ssc_factor) @ XtX_inv
    else:
        V = sigma2 * XtX_inv

    # ── Degrees of freedom for t-tests ──────────────────────────────────
    # fixest uses G - 1 where G = number of entity clusters in the
    # estimation sample (after removing never-treated obs).
    if cluster is not None:
        cluster_arr = data[cluster].values[keep_mask]
        n_clusters = len(np.unique(cluster_arr))
        df_test = max(n_clusters - 1, 1)
    else:
        df_test = df_resid

    # ── ATT: time::0 period-level aggregate ──────────────────────────────
    # In R's sunab, the ATT is the cohort-weighted average of the
    # time::0 (rel_period=0) interaction coefficients, where weights
    # are cohort shares within that period.
    shares = _compute_sunab_shares(D, dummy_cols_in_D)
    att_coefs = beta[dummy_keep_idx]
    att = float(np.sum(shares * att_coefs))

    # ATT SE: sqrt(w' V w) with weights only on interaction dummies
    w = np.zeros(k)
    w[dummy_keep_idx] = shares
    att_var = float(w @ V @ w)
    att_se = np.sqrt(max(att_var, 0.0))

    # ── Period-level and cohort-level aggregated views ────────────────────
    (period_coefs, period_ses, period_names,
     cohort_coefs, cohort_ses, cohort_names,
    ) = _compute_sunab_aggregates(
        kept_names, beta, V, D, dummy_names, dummy_keep_idx,
    )

    # Override ATT with the time::0 period-level aggregate
    t0_idx = period_names.index("time::0") if "time::0" in period_names else None
    if t0_idx is not None:
        att = float(period_coefs[t0_idx])
        att_se = float(period_ses[t0_idx])

    # ── t-stat and p-value for ATT ──────────────────────────────────────
    from scipy.stats import t as _t_dist
    att_t = att / att_se if att_se > 0 else np.nan
    att_p = 2.0 * _t_dist.sf(np.abs(att_t), df=df_test) if att_se > 0 else np.nan

    # ── SEs, t-stats, p-values for all coefficients ─────────────────────
    se_all = np.sqrt(np.maximum(np.diag(V), 0.0))
    t_all = np.where(se_all > 0, beta / se_all, np.nan)
    p_all = 2.0 * _t_dist.sf(np.abs(t_all), df=df_test)

    t_crit = _t_dist.ppf(0.975, df=df_test)
    ci_lower = beta - t_crit * se_all
    ci_upper = beta + t_crit * se_all

    # ── R-squared (fixest convention: SST over ALL observations) ────────
    sst_all = float(np.sum((y_arr - np.mean(y_arr)) ** 2, dtype=float))
    # R² = 1 - sigma2 * (n - nparams) / (SST_all / (n - 1))
    #    = 1 - SSR * (n - 1) / (SST_all * (n - nparams))
    r_squared = (
        1.0 - sigma2 * (n - nparams) / (sst_all / max(n - 1, 1))
        if sst_all > 0 else 0.0
    )
    adj_r_squared = (
        1.0 - (1.0 - r_squared) * (n - 1) / max(n - nparams, 1)
        if sst_all > 0 else 0.0
    )

    # ── Build coefficient series (post-collinearity, non-collinear set) ─
    coef_series = pd.Series(beta, index=kept_names)
    se_series = pd.Series(se_all, index=kept_names)
    t_series = pd.Series(t_all, index=kept_names)
    p_series = pd.Series(p_all, index=kept_names)
    conf_int = pd.DataFrame(
        {"lower": ci_lower, "upper": ci_upper},
        index=kept_names,
    )

    return SaDiDResult(
        cohort_var=cohort,
        period_var=period,
        ref_period=ref_period,
        nobs=n_total,
        dep_var=y,
        coefficients=coef_series,
        std_errors=se_series,
        t_stats=t_series,
        p_values=p_series,
        conf_int=conf_int,
        att=att,
        att_se=att_se,
        att_t_stat=att_t,
        att_p_value=att_p,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        sigma2=sigma2,
        vce=V,
        call=call,
        period_coefs=period_coefs,
        period_ses=period_ses,
        period_names=period_names,
        cohort_coefs=cohort_coefs,
        cohort_ses=cohort_ses,
        cohort_names=cohort_names,
    )
