"""Linear-in-parameters GMM estimator (public API).

This wraps the shared solver :func:`open_econs.models._gmm_core.estimate_gmm`
with the library's standard IV formula grammar (reused from
:func:`open_econs.models.linear.iv.iv`) and result conventions.

Scope: linear-in-parameters GMM only.  The moment/weighting matrices ``Y``,
``X``, ``Z`` are built from the formula; the one-step weighting is the identity
(plain/iid, ``(Z'Z)^{-1}``) so this is the textbook GMM estimator rather than
the Arellano-Bond panel-difference variant.  No nonlinear or
substitutable-expression moment conditions are supported.

The AB-specific conventions (``sig2_scale``, ``small_sample_correction``) are
kept internal to :func:`open_econs.models.linear.abond.abond` and are never
exposed here -- this estimator always uses the generic defaults.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from open_econs._version import __version__
from open_econs._internal import errors
from open_econs.core.base import BaseModel
from open_econs.core.call_capture import capture_call as _capture_call
from open_econs.models._gmm_core import estimate_gmm as _estimate_gmm
from open_econs.models.linear.iv import _parse_iv_formula


class GMMResult(BaseModel):
    """Result of a linear-in-parameters GMM estimation.

    Lean by design: coefficients, standard errors, the variance-covariance
    matrix (``.vcov()``), and the Hansen J overidentification test.  No
    AR(1)/AR(2) serial-correlation tests -- those are panel-difference specific
    and live in :class:`open_econs.core.panel_results.ArellanoBondResult`.
    """

    def __init__(
        self,
        *,
        formula: str,
        coefficients: pd.Series,
        std_errors: pd.Series,
        z_stats: pd.Series,
        p_values: pd.Series,
        conf_int: pd.DataFrame,
        step: str,
        cov_type: str,
        n_obs: int,
        n_instruments: int,
        hansen_j: float,
        hansen_j_pvalue: float,
        hansen_j_dof: int,
        vcov_matrix: np.ndarray,
        call: dict[str, Any],
    ) -> None:
        self.formula = formula
        self.data_shape = (n_obs, len(coefficients))
        self.cov_type = cov_type
        self.call = call
        self.timestamp = datetime.now()
        self.package_version = __version__

        self.coefficients = coefficients
        self.std_errors = std_errors
        self.z_stats = z_stats
        self.p_values = p_values
        self.conf_int = conf_int
        self.step = step
        self.n_obs = n_obs
        self.n_instruments = n_instruments
        self.hansen_j = hansen_j
        self.hansen_j_pvalue = hansen_j_pvalue
        self.hansen_j_dof = hansen_j_dof
        self._vcov = np.asarray(vcov_matrix, dtype=float)

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
        header = (
            f"                    Linear GMM Regression Results                       \n"
            f"======================================================================\n"
            f"Dep. Variable:               {self.formula.split('~')[0].strip()}\n"
            f"Step:                        {self.step}\n"
            f"No. Observations:            {self.n_obs}\n"
            f"No. Instruments (L):         {self.n_instruments}\n"
            f"Covariance Type:             {self.cov_type}\n"
            f"Hansen J:                    {self.hansen_j:.4f} (df={self.hansen_j_dof}, "
            f"p={self.hansen_j_pvalue:.4f})\n"
            f"======================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return header + tbl + "\n======================================================================\n"

    def vcov(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._vcov,
            index=self.coefficients.index,
            columns=self.coefficients.index,
        )

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["step"] = self.step
        d["hansen_j"] = self.hansen_j
        d["hansen_j_pvalue"] = self.hansen_j_pvalue
        d["hansen_j_dof"] = self.hansen_j_dof
        d["n_instruments"] = self.n_instruments
        return d


def gmm(
    formula: str,
    data: pd.DataFrame,
    *,
    step: str = "twostep",
    cov_type: str = "robust",
    cluster: str | None = None,
) -> GMMResult:
    """Estimate a linear-in-parameters GMM regression.

    Parameters
    ----------
    formula : str
        IV/2SLS grammar ``y ~ exog | endog ~ instruments``.  Variables left of
        ``|`` (outside the inner ``~``) are exogenous controls; those left of
        the inner ``~`` are endogenous regressors; those to the right of the
        inner ``~`` are the instruments.  The simpler legacy form
        ``y ~ rhs | instruments`` is accepted (treating all RHS variables as
        endogenous) for backward compatibility with :func:`iv`.
    data : pd.DataFrame
        Data containing all variables referenced in *formula*.
    step : {"onestep", "twostep"}, default "twostep"
        GMM step.  The one-step estimator with identity weighting is identical
        to 2SLS; the two-step estimator uses the efficient
        ``S = Σ (Zᵢ'eᵢ)(Zᵢ'eᵢ)'`` weighting.
    cov_type : {"robust", "cluster"}, default "robust"
        Covariance estimator.  ``"robust"`` is the heteroskedasticity-robust
        (White) sandwich -- this library's ``cov_type="robust"`` convention, not
        Stata's ``wmatrix()`` naming.  ``"cluster"`` clusters the sandwich by
        the variable named in *cluster*.
    cluster : str, optional
        Column name of the cluster/group variable.  Required when
        ``cov_type="cluster"``; ignored otherwise.

    Returns
    -------
    GMMResult
        Immutable result with coefficients, SEs, ``.vcov()``, and the Hansen J
        overidentification test.

    Notes
    -----
    This estimator uses the shared core's *generic* conventions (``sig2_scale``
    default 1.0, no small-sample correction).  The Arellano-Bond-specific
    normalization is intentionally not available here.
    """
    if step not in ("onestep", "twostep"):
        raise ValueError("step must be 'onestep' or 'twostep'.")
    if cov_type not in ("robust", "cluster"):
        raise ValueError("cov_type must be 'robust' or 'cluster'.")
    if cov_type == "cluster":
        if cluster is None:
            raise ValueError("cluster= must be provided when cov_type='cluster'.")
        if cluster not in data.columns:
            raise errors.cluster_column_error(cluster, data.columns.tolist())
    elif cluster is not None:
        raise ValueError("cluster= is only used when cov_type='cluster'.")

    call = _capture_call(
        formula=formula, step=step, cov_type=cov_type, cluster=cluster,
    )

    parsed = _parse_iv_formula(formula, data)
    Y = parsed["y"]
    X = parsed["X"]
    instr_matrix = parsed["instr_matrix"]
    exog_idx = parsed["exog_idx"]
    coef_names = parsed["coef_names"]
    n = len(Y)

    # Full instrument matrix Z = [exogenous regressors (incl. intercept),
    # explicit instruments].  The exogenous regressors are their own instruments;
    # the endogenous regressors are instrumented by `instr_matrix`.
    z_parts = [X[:, exog_idx]] if exog_idx else []
    if instr_matrix.shape[1] > 0:
        z_parts.append(instr_matrix)
    if not z_parts:
        raise ValueError(
            "No instruments available; a GMM estimator needs at least as many "
            "instruments as regressors."
        )
    Z = np.column_stack(z_parts)

    if cov_type == "cluster":
        eq_entity = data.loc[parsed["index"], cluster].values
        robust = True
    else:
        eq_entity = np.arange(n)
        robust = True

    # Map the public step keyword onto the core's hyphenated spelling.
    core_step = "one-step" if step == "onestep" else "two-step"

    # Generic defaults (sig2_scale=1.0, small_sample_correction=False) are used;
    # the AB-specific conventions are intentionally not exposed here.
    est = _estimate_gmm(Y, X, Z, eq_entity, core_step, robust=robust)

    coefficients = pd.Series(est["b"], index=coef_names)
    std_errors = pd.Series(est["se"], index=coef_names)
    z_stats = pd.Series(
        np.where(est["se"] > 0, est["b"] / est["se"], np.nan), index=coef_names,
    )
    p_values = pd.Series(
        2.0 * (1.0 - _norm.cdf(np.abs(z_stats.values))), index=coef_names,
    )
    conf_int = pd.DataFrame(
        {
            "lower": est["b"] - 1.96 * est["se"],
            "upper": est["b"] + 1.96 * est["se"],
        },
        index=coef_names,
    )

    return GMMResult(
        formula=formula,
        coefficients=coefficients,
        std_errors=std_errors,
        z_stats=z_stats,
        p_values=p_values,
        conf_int=conf_int,
        step=step,
        cov_type=cov_type,
        n_obs=int(n),
        n_instruments=int(Z.shape[1]),
        hansen_j=float(est["J"]),
        hansen_j_pvalue=float(est["p_j"]),
        hansen_j_dof=int(est["dof_j"]),
        vcov_matrix=est["V"],
        call=call,
    )
