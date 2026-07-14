"""Gardner (2022) Two-Stage Difference-in-Differences estimator.

Implements the DID2S estimator from:
    Gardner, John. 2022. "Two-Stage Differences in Differences."
    arXiv:2207.05943. Working paper (May 2024 revision adds Thakral,
    To, Yap as co-authors — implement against original single-author
    formulation).

The estimator performs:
    1. First stage: regress outcome on covariates/FEs using only
       untreated (never-treated or pre-treatment) observations.
    2. Second stage: regress first-stage residuals on treatment
       indicators using all observations.
    3. ATT = coefficient on treatment indicator in second stage.
    4. Cluster-robust SEs via influence functions.

R parity anchor: ``did2s::did2s()`` (v1.2.1).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from open_econs._version import __version__
from open_econs.core.call_capture import capture_call as _capture_call
from open_econs.core.base import BaseModel


class GardnerResult(BaseModel):
    """Result of Gardner (2022) Two-Stage Difference-in-Differences.

    Immutable result exposing a uniform interface: ``.tidy()`` (coefficients,
    SEs, t-stats, p-values, CI), ``.summary()`` (text), ``.export()``
    (CSV/JSON/Pickle), ``.vcov()``.  The key quantity is the ATT estimate
    ``att`` with its ``att_se`` / ``att_t_stat`` / ``att_p_value``.

    References
    ----------
    Gardner, John. 2022. "Two-Stage Differences in Differences."
    arXiv:2207.05943. Working paper.
    """
    def __init__(
        self,
        *,
        formula_first_stage: str,
        formula_second_stage: str,
        nobs: int,
        dep_var: str,
        treatment_var: str,
        cluster_var: str | None,
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
        call: dict[str, Any],
    ) -> None:
        self.formula_first_stage = formula_first_stage
        self.formula_second_stage = formula_second_stage
        self.data_shape = (nobs, coefficients.shape[0])
        self.call = call
        self.timestamp = datetime.now()
        self.package_version = __version__

        self.nobs = nobs
        self.dep_var = dep_var
        self.treatment_var = treatment_var
        self.cluster_var = cluster_var
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
            f"            Gardner (2022) Two-Stage DID Results             \n"
            f"============================================================\n"
            f"Dep. Variable:               {self.dep_var}\n"
            f"Treatment variable:          {self.treatment_var}\n"
            f"Cluster:                     {self.cluster_var if self.cluster_var else 'none'}\n"
            f"No. Observations:            {self.nobs}\n"
            f"R-squared:                   {self.r_squared:.6f}\n"
            f"Adj. R-squared:              {self.adj_r_squared:.6f}\n"
            f"Sigma2:                      {self.sigma2:.6f}\n"
            f"============================================================\n"
            f"First stage:  {self.formula_first_stage}\n"
            f"Second stage: {self.formula_second_stage}\n"
            f"============================================================\n"
            f"ATT ({self.treatment_var}):\n"
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
            np.outer(self.std_errors.values, self.std_errors.values),
            index=self.coefficients.index,
            columns=self.coefficients.index,
        )


def _gardner_two_stage_vce(
    X1: np.ndarray,
    X2: np.ndarray,
    first_u: np.ndarray,
    second_u: np.ndarray,
    treat_mask: np.ndarray,
    cluster: np.ndarray,
) -> np.ndarray:
    """Two-stage cluster-robust VCE matching did2s::did2s().

    Computes the influence function for the Gardner DID2S estimator:
        IF = IF_fs - IF_ss
    where:
        IF_ss = (X2'X2)^{-1} X2' second_u   (second-stage OLS IF)
        IF_fs = (X2'X2)^{-1} gamma' X10' first_u  (first-stage IF)
        gamma = (X10'X1)^{-1} X10' X2  (cross-regression coefficient)
        X10 = X1 with treated rows zeroed out

    Cluster-robust VCE:
        V = sum_g (IF_g)(IF_g')  where IF_g = sum of IFs in cluster g
    """
    from numpy.linalg import inv

    n, k2 = X2.shape
    k1 = X1.shape[1]

    # (X2'X2)^{-1}
    XtX2_inv = inv(X2.T @ X2)

    # X10: X1 with treated rows zeroed out
    X10 = X1.copy()
    X10[treat_mask] = 0.0

    # gamma = (X10'X10)^{-1} (X1'X2)
    # NOTE: R's did2s uses Matrix::crossprod(x1, x2) — the ORIGINAL x1,
    # not x10. The cross-product x1'x2 is over ALL observations.
    # Only the left side uses x10'x10 (untreated only).
    X10tX10 = X10.T @ X10
    gamma = inv(X10tX10) @ (X1.T @ X2)

    # Per-observation IFs: k2 x n matrix
    # IF_ss_j = XtX2_inv @ (X2[j] * second_u[j])
    # IF_fs_j = XtX2_inv @ gamma' @ (X10[j] * first_u[j])
    IF_ss = XtX2_inv @ (X2.T * second_u[np.newaxis, :])  # k2 x n
    IF_fs = XtX2_inv @ gamma.T @ (X10.T * first_u[np.newaxis, :])  # k2 x n
    IF = IF_fs - IF_ss  # k2 x n

    # Cluster-robust VCE: sum over clusters of (sum of IFs in cluster)^2
    unique_clusters = np.unique(cluster)
    V = np.zeros((k2, k2))
    for c in unique_clusters:
        idx = cluster == c
        IF_c = IF[:, idx].sum(axis=1)  # k2 vector
        V += np.outer(IF_c, IF_c)

    return V


def did_gardner(
    data: pd.DataFrame,
    y: str,
    first_stage: str,
    second_stage: str,
    treatment: str,
    cluster: str | None = None,
) -> GardnerResult:
    """Gardner (2022) Two-Stage Difference-in-Differences estimator.

    Implements the DID2S estimator: a two-stage procedure where (1) the
    outcome is regressed on covariates/FEs using only untreated
    observations, and (2) the first-stage residuals are regressed on
    treatment indicators using all observations.  The coefficient on the
    treatment indicator is the ATT.

    Cluster-robust standard errors are computed via influence functions,
    matching ``did2s::did2s()`` in R.

    Parameters
    ----------
    data : pd.DataFrame
        Analysis data.  Must contain columns for ``y``, the treatment
        indicator, and any variables in ``first_stage``/``second_stage``.
    y : str
        Name of the outcome variable.
    first_stage : str
        RHS formula for the first-stage regression (estimated on
        untreated observations only).  Use ``"0 + factor(entity) +
        factor(time)"`` for entity and time fixed effects.
    second_stage : str
        RHS formula for the second-stage regression (estimated on all
        observations).  Typically just the treatment indicator name.
    treatment : str
        Name of the binary treatment indicator column (1 = treated).
    cluster : str, optional
        Column name for cluster-robust standard errors.

    Returns
    -------
    GardnerResult
        Immutable result with ``.tidy()``, ``.summary()``, ``.vcov()``.

    References
    ----------
    Gardner, John. 2022. "Two-Stage Differences in Differences."
    arXiv:2207.05943. Working paper.
    """
    call = _capture_call(
        y=y,
        first_stage=first_stage,
        second_stage=second_stage,
        treatment=treatment,
        cluster=cluster,
    )

    if treatment not in data.columns:
        raise ValueError(
            f"Treatment column '{treatment}' not found in data. "
            f"Available: {list(data.columns)}"
        )

    if cluster is not None and cluster not in data.columns:
        raise ValueError(
            f"Cluster column '{cluster}' not found in data. "
            f"Available: {list(data.columns)}"
        )

    # Identify treated observations
    treat_mask = data[treatment].astype(bool).values
    untreat_mask = ~treat_mask

    n_total = len(data)
    n_untreated = int(untreat_mask.sum())

    if n_untreated == 0:
        raise ValueError("No untreated observations found.")

    # ---- Build design matrices ----
    # First stage: on untreated only
    from formulaic import Formula
    y_arr = data[y].values.astype(float)

    # First stage design matrix (untreated only)
    fs_formula = Formula(f"~ 0 + {first_stage}")
    fs_spec = fs_formula.get_model_matrix(data.loc[untreat_mask])
    X1 = fs_spec.values.astype(float) if hasattr(fs_spec, "values") else np.asarray(fs_spec)
    y1 = y_arr[untreat_mask]

    # Fit first stage
    from numpy.linalg import lstsq
    beta1, _, _, _ = lstsq(X1, y1, rcond=None)
    fitted1 = X1 @ beta1

    # Predict for ALL observations
    # Need to build X1_full for all observations
    fs_spec_full = fs_formula.get_model_matrix(data)
    X1_full = fs_spec_full.values.astype(float) if hasattr(fs_spec_full, "values") else np.asarray(fs_spec_full)
    fitted_full = X1_full @ beta1

    # First-stage residuals for all observations
    first_u = y_arr - fitted_full

    # Second stage design matrix (all observations)
    ss_formula = Formula(f"~ 0 + {second_stage}")
    ss_spec = ss_formula.get_model_matrix(data)
    X2 = ss_spec.values.astype(float) if hasattr(ss_spec, "values") else np.asarray(ss_spec)

    # Second-stage estimation (on all observations with first-stage residuals)
    n, k = X2.shape
    from numpy.linalg import inv
    XtX2 = X2.T @ X2
    XtX2_inv = inv(XtX2)
    beta2 = XtX2_inv @ (X2.T @ first_u)
    second_u = first_u - X2 @ beta2

    # Coefficients and SEs
    sigma2 = float(np.sum(second_u ** 2) / max(n - k, 1))

    if cluster is not None:
        cluster_arr = data[cluster].values
        V = _gardner_two_stage_vce(
            X1_full, X2, first_u, second_u, treat_mask, cluster_arr,
        )
    else:
        V = sigma2 * XtX2_inv

    se2 = np.sqrt(np.maximum(np.diag(V), 0.0))

    # t-stats and p-values
    from scipy.stats import norm as _norm, t as _t_dist
    df_r = n - k
    t_arr = np.where(se2 > 0, beta2 / se2, np.nan)
    p_arr = 2.0 * _t_dist.sf(np.abs(t_arr), df=df_r)

    # Confidence intervals
    t_crit = _t_dist.ppf(0.975, df=df_r)
    ci_lower = beta2 - t_crit * se2
    ci_upper = beta2 + t_crit * se2

    # R-squared from first stage (on untreated)
    ss_resid_1 = float(np.sum((y1 - fitted1) ** 2))
    ss_tot_1 = float(np.sum((y1 - np.mean(y1)) ** 2))
    r_squared_1 = 1.0 - ss_resid_1 / ss_tot_1 if ss_tot_1 > 0 else 0.0

    # R-squared from second stage
    ss_resid_2 = float(np.sum(second_u ** 2))
    ss_tot_2 = float(np.sum((first_u - np.mean(first_u)) ** 2))
    r_squared_2 = 1.0 - ss_resid_2 / ss_tot_2 if ss_tot_2 > 0 else 0.0
    adj_r_squared_2 = 1.0 - (ss_resid_2 / max(n - k, 1)) / (ss_tot_2 / max(n - 1, 1)) if ss_tot_2 > 0 else 0.0

    # Extract ATT (coefficient on treatment)
    coef_names = list(ss_spec.columns) if hasattr(ss_spec, "columns") else [f"V{i}" for i in range(k)]
    coef_series = pd.Series(beta2, index=coef_names)
    se_series = pd.Series(se2, index=coef_names)
    t_series = pd.Series(t_arr, index=coef_names)
    p_series = pd.Series(p_arr, index=coef_names)
    conf_int = pd.DataFrame({"lower": ci_lower, "upper": ci_upper}, index=coef_names)

    # ATT is the coefficient on the treatment indicator
    # Find it in the design matrix columns
    treat_col_idx = None
    for i, name in enumerate(coef_names):
        if name == treatment or name.endswith(f":{treatment}") or name.startswith(f"{treatment}:"):
            treat_col_idx = i
            break
    if treat_col_idx is None:
        # If treatment not found by name, assume it's the only coefficient
        treat_col_idx = 0

    att = float(beta2[treat_col_idx])
    att_se = float(se2[treat_col_idx])
    att_t = float(t_arr[treat_col_idx])
    att_p = float(p_arr[treat_col_idx])

    return GardnerResult(
        formula_first_stage=first_stage,
        formula_second_stage=second_stage,
        nobs=n_total,
        dep_var=y,
        treatment_var=treatment,
        cluster_var=cluster,
        coefficients=coef_series,
        std_errors=se_series,
        t_stats=t_series,
        p_values=p_series,
        conf_int=conf_int,
        att=att,
        att_se=att_se,
        att_t_stat=att_t,
        att_p_value=att_p,
        r_squared=r_squared_2,
        adj_r_squared=adj_r_squared_2,
        sigma2=sigma2,
        call=call,
    )
