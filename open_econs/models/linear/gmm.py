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
from open_econs.core.cov_type import validate_cov_type
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
        """R-broom-style coefficient table (Variable, Coef, Std Err, z, P>|z|, CI)."""
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
        """Pretty-printed terminal summary of the GMM fit (incl. Hansen J test)."""
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
        """Return the GMM parameter variance-covariance matrix as a DataFrame."""
        return pd.DataFrame(
            self._vcov,
            index=self.coefficients.index,
            columns=self.coefficients.index,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable dict; extends the base payload with GMM-specific fields."""
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
    step: str = "two-step",
    cov_type: str = "robust",
    cluster: str | None = None,
    lags: int | None = None,
    time: str | None = None,
    hac_adjust: bool = False,
    windmeijer: bool = True,
    robust_meat: str = "one-step",
    weight: str = "stata",
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
    step : {"one-step", "two-step"}, default "two-step"
        GMM step.  The one-step estimator with identity weighting is identical
        to 2SLS; the two-step estimator uses the efficient
        ``S = Σ (Zᵢ'eᵢ)(Zᵢ'eᵢ)'`` weighting.
    cov_type : {"robust", "cluster", "HAC"}, default "robust"
        Covariance estimator.  ``"robust"`` is the heteroskedasticity-robust
        (White) sandwich -- this library's ``cov_type="robust"`` convention, not
        Stata's ``wmatrix()`` naming.  ``"cluster"`` clusters the sandwich by
        the variable named in *cluster*.  ``"HAC"`` uses Newey-West (1987)
        autocorrelation-robust standard errors and, in two-step GMM, a
        HAC-weighted efficient weighting matrix (Hansen 1982, Newey & West
        1994).  Requires ``lags`` and ``time`` parameters.  ``"cluster"`` takes
        precedence over ``"HAC"`` (if both ``cluster`` and ``cov_type="HAC"``
        are given, cluster-robust SEs are used).
    cluster : str, optional
        Column name of the cluster/group variable.  Required when
        ``cov_type="cluster"``; ignored otherwise.  When ``cov_type="HAC"``,
        *cluster* names the entity grouping for the per-entity time-series HAC
        computation (the Newey-West estimator is computed within each entity
        cluster and accumulated).  If omitted when ``cov_type="HAC"``, each
        observation is treated as its own entity (HAC across the
        full time-ordered sample).
    lags : int, optional
        Number of Newey-West lags (bandwidth).  Required when
        ``cov_type="HAC"``.
    time : str, optional
        Column name of the time variable (used for time ordering within each
        entity when determining the HAC autocorrelation structure).  Required
        when ``cov_type="HAC"``.
    hac_adjust : bool, default False
        If True, apply the ``N/(N-K)`` degrees-of-freedom correction to the
        HAC long-run variance matrix ``S`` (analogous to the adjustment
        applied to ``newey_west_cov`` in OLS contexts).  For exactly-identified
        GMM this produces the same SEs as ``ols()``/``iv()`` with
        ``hac_adjust=True``.
    windmeijer : bool, default True
        If True (default), apply the Windmeijer (2005) finite-sample correction
        to the two-step robust VCE.  This is the recommended practice in the
        econometric literature and matches Stata's ``xtabond``/``xtdpd``
        default.  If False, skip the correction, reproducing Stata's ``gmm``
        command default (which does NOT apply Windmeijer — confirmed via
        gmm.ado source; contrast ``xtabond``/``xtdpd`` which DO apply it).
        Ignored when ``step="one-step"`` or ``cov_type`` is not ``"robust"``.
    robust_meat : {"one-step", "two-step"}, default "one-step"
        Which residuals feed the robust **MEAT** of the two-step VCE sandwich.
        The literature and R's ``gmm`` package (``vcov="MDS"``) build the
        robust meat from the **one-step** residuals ``e1``; Stata's ``gmm``
        command builds it from the **two-step** residuals ``e2``.  Set
        ``robust_meat="two-step"`` (together with ``windmeijer=False``) to
        reproduce Stata's ``gmm`` two-step robust VCE exactly (verified
        against Stata's extracted ``e(S)`` matrix).  IMPORTANT: this controls
        ONLY the robust meat; the efficient-weight bread stays at the
        one-step residuals.  For exact Stata ``gmm`` parity, use BOTH
        ``robust_meat="two-step"`` and ``windmeijer=False``; setting only one
        of them yields a hybrid that matches neither R/literature nor Stata.
        Ignored when ``step="one-step"`` or ``cov_type``
        is not ``"robust"``.
    weight : {"stata", "iid"}, default "stata"
        Which covariance structure feeds the two-step efficient-weight BREAD.
        The default ``"stata"`` uses the same structure as the VCE (cluster S
        for ``cov_type="cluster"``, HAC S for ``cov_type="HAC"``, iid S for
        ``cov_type="robust"``), matching Stata's ``gmm`` command and making the
        two-step coefficient change across robust/cluster/HAC.  ``"iid"``
        instead always uses the plain heteroskedasticity-robust iid weight
        (each observation its own group), while the VCE meat keeps the
        cov-structure S — matching R's ``gmm`` package for cluster/HAC.  Set
        ``weight="iid"`` together with ``robust_meat="two-step"`` and
        ``windmeijer=False`` to reproduce R's ``gmm(..., vcov="iid",
        cluster=)`` coefficient and cluster-robust SEs.  This flag governs ONLY
        the efficient weight; the meat is governed by ``robust_meat``.  See
        GMM-RCLUSTER in ``FUTURE_WORK.md`` and ``methodology/linear/gmm.md``.

    Returns
    -------
    GMMResult
        Immutable result with coefficients, SEs, ``.vcov()``, and the Hansen J
        overidentification test.

    Notes
    -----
    This estimator uses the shared core's *generic* conventions (``sig2_scale``
    default 1.0, no small-sample correction).  The Arellano-Bond-specific
    normalization is intentionally not available here.  For dynamic panel
    models with lagged dependent variables and instrument construction from
    the panel structure, see :func:`abond`; this function provides plain
    linear GMM only (no panel-specific instrument or lag handling).

    Convention notes (source-confirmed):
      * **Intercept as instrument.**  ``gmm()`` always includes the intercept
        as its own instrument in ``Z``.  For the formula
        ``y ~ x1 + x2 | z1 + z2``, ``Z = [intercept, z1, z2]`` (3
        instruments for 3 parameters in the exactly-identified case).
        Stata's ``gmm`` command must replicate this by including
        ``1*(y - Xb)`` as an explicit moment condition.
      * **One-step J.**  The one-step non-robust J uses the model-based
        weighting ``A1 = (Z'Z)^{-1} / sig2``, giving
        ``J = g'(Z'Z)^{-1}g / sig2``.  This matches R's
        ``gmm::specTest(tsls)`` but differs from Stata's ``e(J)`` which
        uses the robust sandwich S matrix.  Both are valid; see
        :mod:`open_econs.models._gmm_core` for the full derivation.
      * **Windmeijer correction.**  For two-step GMM with
        ``cov_type="robust"``, the ``windmeijer`` flag controls whether the
        Windmeijer (2005) finite-sample correction is applied to the VCE.
        Default ``windmeijer=True`` matches the econometric literature's
        recommended practice and Stata's ``xtabond``/``xtdpd``.  Setting
        ``windmeijer=False`` reproduces Stata's ``gmm`` command, which does
        NOT apply the correction (confirmed via gmm.ado source).  See
        GMM-WC in ``FUTURE_WORK.md``.
    """
    if step not in ("one-step", "two-step"):
        raise ValueError("step must be 'one-step' or 'two-step'.")
    cov_type = validate_cov_type(
        cov_type,
        accepted={"robust", "cluster", "HAC"},
        estimator="gmm()",
    )
    if cov_type == "cluster":
        if cluster is None:
            raise ValueError("cluster= must be provided when cov_type='cluster'.")
        if cluster not in data.columns:
            raise errors.cluster_column_error(cluster, data.columns.tolist())
    elif cluster is not None and cov_type != "HAC":
        raise ValueError("cluster= is only used when cov_type='cluster'.")
    if cov_type == "HAC":
        if lags is None:
            raise ValueError("lags= must be provided when cov_type='HAC'.")
        if time is None:
            raise ValueError("time= must be provided when cov_type='HAC'.")
        if time not in data.columns:
            raise errors.missing_column_error(time, data.columns.tolist())

    if weight not in ("stata", "iid"):
        raise ValueError("weight must be 'stata' or 'iid'.")
    call = _capture_call(
        formula=formula, step=step, cov_type=cov_type, cluster=cluster,
        lags=lags, time=time, hac_adjust=hac_adjust, windmeijer=windmeijer,
        robust_meat=robust_meat, weight=weight,
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

    # Public step spelling now matches the core's hyphenated spelling.
    core_step = step

    hac_time_labels: np.ndarray | None = None
    hac_max_lags: int | None = None
    use_hac_adjust = False
    if cov_type == "HAC":
        hac_time_labels = data.loc[parsed["index"], time].values
        hac_max_lags = lags
        use_hac_adjust = hac_adjust
        # Use cluster groups as entity dimension for per-entity HAC if provided
        if cluster is not None:
            if cluster not in data.columns:
                raise errors.cluster_column_error(cluster, data.columns.tolist())
            eq_entity = data.loc[parsed["index"], cluster].values
        else:
            eq_entity = np.arange(n)
    elif cov_type == "cluster":
        eq_entity = data.loc[parsed["index"], cluster].values
    else:
        eq_entity = np.arange(n)

    robust = True

    # Generic defaults (sig2_scale=1.0, small_sample_correction=False) are used;
    # the AB-specific conventions are intentionally not exposed here.
    est = _estimate_gmm(
        Y, X, Z, eq_entity, core_step, robust=robust,
        time_labels=hac_time_labels, max_lags=hac_max_lags,
        hac_adjust=use_hac_adjust, windmeijer=windmeijer,
        robust_meat=robust_meat, weight=weight,
    )

    if cov_type == "HAC":
        cov_type = f"HAC({lags})"

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
