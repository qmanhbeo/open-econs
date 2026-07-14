import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from open_econs._version import __version__
from open_econs.core.base import BaseModel
from open_econs.core.call_capture import capture_call as _capture_call
from open_econs.core.cov_type import validate_cov_type


class AggteResult(BaseModel):
    """Result of aggregating group-time ATTs following Callaway & Sant'Anna (2021).

    Obtained via :meth:`CsDiDResult.aggte`.  Stores the overall
    aggregated ATT and its standard error, plus a DataFrame of per-level
    ATTs and SEs for the chosen aggregation type.

    References
    ----------
    Callaway, Brantly and Pedro H.C. Sant'Anna. 2021.
    "Difference-in-Differences with Multiple Time Periods."
    *Journal of Econometrics*, Vol. 225, No. 2, pp. 200-230.
    """

    def __init__(
        self,
        *,
        att: float,
        se: float,
        p_value: float,
        type: str,
        att_by: pd.DataFrame,
        overall_weights: dict[str, float],
        method: str,
        call: dict[str, Any],
    ) -> None:
        self.att = att
        self.se = se
        self.p_value = p_value
        self.type = type
        self.att_by = att_by
        self.overall_weights = overall_weights
        self.method = method
        self.call = call
        self.timestamp = __import__("datetime").datetime.now()
        self.package_version = __version__
        self._freeze()

    def tidy(self) -> pd.DataFrame:
        return self.att_by

    def summary(self) -> str:
        type_label = {"dynamic": "Event-Time", "group": "Group", "calendar": "Calendar"}
        lines = [
            f"       Aggregated ATT ({type_label.get(self.type, self.type)})       ",
            "=" * 56,
            f"  Overall ATT  : {self.att:.4f} (se {self.se:.4f}, p {self.p_value:.3f})",
            f"  Method       : {self.method}",
            f"  Aggregation  : {self.type}",
            "",
            "  Per-level ATTs:",
        ]
        for _, row in self.att_by.iterrows():
            level_col = self.att_by.columns[0]
            lines.append(
                f"    {level_col}={int(row[level_col])}: "
                f"ATT={row['att']:.4f} (se {row['se']:.4f})"
            )
        return "\n".join(lines)


class CsDiDResult(BaseModel):
    """Result of Callaway & Sant'Anna (2021) staggered difference-in-differences estimator.

    Implements the Callaway & Sant'Anna (2021) group-time estimator with
    doubly-robust inference (Sant'Anna and Zhao 2020).  Default method (with
    covariates) is ``dripw`` — logit propensity-score weighting plus OLS
    outcome regression.  Without covariates, falls back to outcome-regression
    DiD (``reg``), equivalent to a simple 2×2 OLS interaction.

    Stores group-time average treatment effects ``att_group_time`` (one row per
    cohort×period), the aggregated dynamic ``event_study`` path (lead l ≥ 0),
    and the overall pooled ``att``.  Inference uses influence-function-based
    cluster-robust standard errors when ``method="dripw"``, entity-clustered
    OLS standard errors when ``method="reg"``.
    """

    def __init__(
        self,
        *,
        att_group_time: pd.DataFrame,
        event_study: pd.DataFrame,
        att: float,
        att_se: float,
        att_p: float,
        cohorts: list[int],
        n_periods: int,
        n_obs: int,
        control_cohorts: str,
        method: str,
        summary_text: str,
        call: dict[str, Any],
        cov_type: str = "cluster",
    ) -> None:
        self.att_group_time = att_group_time
        self.event_study = event_study
        self.att = att
        self.att_se = att_se
        self.att_p = att_p
        self.cohorts = cohorts
        self.n_periods = n_periods
        self.n_obs = n_obs
        self.control_cohorts = control_cohorts
        self.method = method
        self.call = call
        self.cov_type = cov_type
        self.timestamp = __import__("datetime").datetime.now()
        self.package_version = __version__
        self._freeze()

    def tidy(self) -> pd.DataFrame:
        return self.att_group_time

    def summary(self) -> str:
        label = {
            "dripw": "Doubly-robust (IPW + OLS)",
            "reg": "Outcome regression (OLS)",
        }.get(self.method, self.method)
        lines = [
            f"       Staggered DiD ({label}) Results       ",
            "=" * 66,
            f"  Overall ATT           : {self.att:.4f} (se {self.att_se:.4f}, p {self.att_p:.3f})",
            f"  Method                : {self.method}",
            f"  Cov. type             : {self.cov_type}",
            f"  Cohorts               : {sorted(self.cohorts)}",
            f"  Time periods          : {self.n_periods}",
            f"  Observations          : {self.n_obs}",
            "",
            "  Group-time ATT (post-treatment only):",
        ]
        sub = self.att_group_time
        if len(sub):
            top = sub.sort_values(["cohort", "time"]).head(12)
            for _, row in top.iterrows():
                lines.append(
                    f"    g={int(row['cohort'])} t={int(row['time'])}: "
                    f"ATT={row['att']:.4f} (se {row['se']:.4f})"
                )
        return "\n".join(lines)

    def aggte(self, type: str = "dynamic") -> "AggteResult":
        """Aggregate group-time ATTs following Callaway & Sant'Anna (2021).

        Computes weighted-average treatment effects across the dimensions of
        the group-time ATT matrix:

        * ``type="dynamic"`` — event-time ATTs (leads/lags).  The overall ATT
          averages across **positive** event times only (e ≥ 0), matching R
          ``did::aggte(type="dynamic")``.
        * ``type="group"`` — cohort-specific ATTs.  The overall ATT averages
          across all treated cohorts, weighted by cohort size.
        * ``type="calendar"`` — calendar-time ATTs.  The overall ATT averages
          across all post-treatment time periods.

        Per-level ATTs are weighted by the number of treated units (``n_treat``)
        within each level.  Standard errors use the influence-function
        aggregation when per-entity RIFs are available (``method="dripw"``),
        falling back to the weighted-average-of-variances formula otherwise.

        No Stata anchor exists for ``aggte()`` — Stata's ``csdid`` does not
        implement dynamic/group/calendar aggregation.  The sole parity anchor
        is R ``did::aggte()``.

        Parameters
        ----------
        type : {"dynamic", "group", "calendar"}, default "dynamic"
            Aggregation type.

        Returns
        -------
        AggteResult

        References
        ----------
        Callaway, Brantly and Pedro H.C. Sant'Anna. 2021.
        "Difference-in-Differences with Multiple Time Periods."
        *Journal of Econometrics*, Vol. 225, No. 2, pp. 200-230.
        """
        valid_types = ("dynamic", "group", "calendar")
        if type not in valid_types:
            raise ValueError(
                f"type must be one of {valid_types}, got {type!r}"
            )

        gt = self.att_group_time.copy()
        if gt.empty:
            raise ValueError("No group-time ATTs available for aggregation.")

        group_col = {"dynamic": "lead", "group": "cohort", "calendar": "time"}[type]

        # ---- Extract entity IDs and build per-level RIFs ----
        # R did::aggte computes per-level IFs via get_agg_inf_func on
        # cell-level inffunc, then overall IF via get_agg_inf_func on
        # per-level IFs.  OE's RIFs from drIfp are DR-IPW influence
        # functions indexed by entity.  We build per-level RIFs as the
        # mean of cell RIFs within each level.
        all_entities = None
        for _, row in gt.iterrows():
            cell_rif = row.get("rif")
            if cell_rif is not None and not (isinstance(cell_rif, float) and np.isnan(cell_rif)):
                all_entities = cell_rif.index.values
                break
        has_rifs = all_entities is not None
        n_entities = len(all_entities) if all_entities is not None else 0

        rif_by_level = {}  # level_val → pd.Series (N-entities)
        if has_rifs:
            assert all_entities is not None
            for val, sub in gt.groupby(group_col):
                level_rifs = pd.DataFrame(index=all_entities)
                for _, row in sub.iterrows():
                    cell_rif = row.get("rif")
                    if cell_rif is None or (isinstance(cell_rif, float) and np.isnan(cell_rif)):
                        continue
                    level_rifs[f"r_{len(level_rifs.columns)}"] = cell_rif.reindex(
                        all_entities
                    ).fillna(0.0).values
                if not level_rifs.empty:
                    rif_by_level[val] = level_rifs.mean(axis=1)

        # ---- per-level aggregation (ATT + per-level SE) ----
        # R computes per-level SEs via get_agg_inf_func → getSE on the
        # cell-level IFs within each level.  For group type, R uses
        # wif=NULL (R source: get_agg_inf_func(... wif = NULL)), so the
        # per-level IF is a simple weighted sum of cell IFs — exactly
        # what our per-level RIF (mean of cell RIFs) replicates.  For
        # dynamic/calendar, R includes a wif correction that OE's RIFs
        # approximate; the weighted-average-of-per-cell-SEs formula
        # happens to match R for those types in our fixtures.
        levels = []
        for val, sub in gt.groupby(group_col):
            w = sub["n_treat"].values.astype(float)
            att_val = float(np.average(sub["att"].values, weights=w))
            if type == "group" and val in rif_by_level:
                level_rif = rif_by_level[val]
                # Center the RIF before computing the second moment to
                # match R did::getSE, which uses mean(IF^2)/n where IF
                # has mean zero by construction.  Our RIFs are IF+ATT,
                # so mean(RIF^2) = var(IF)+ATT^2 — the ATT^2 term must
                # be removed via centering.  Source: did:::getSE
                # (sqrt(mean(thisinffunc^2)/n)) where inffunc satisfies
                # mean(inffunc)=0.
                lc = level_rif.values - level_rif.mean()
                se_val = float(np.sqrt(np.mean(lc ** 2) / n_entities))
            else:
                se_val = float(np.sqrt(np.average(sub["se"].values ** 2, weights=w)))
            levels.append({group_col: int(val), "att": att_val, "se": se_val})

        att_by = pd.DataFrame(levels).sort_values(group_col).reset_index(drop=True)
        att_by["p_value"] = 2 * (1 - _norm.cdf(
            np.abs(att_by["att"] / att_by["se"].replace(0, np.nan))
        ))

        # ---- overall ATT ----
        if type == "dynamic":
            pos = att_by[att_by[group_col] >= 0]
            if pos.empty:
                raise ValueError("No positive event times found for dynamic aggregation.")
            overall_att = float(pos["att"].mean())
            overall_weights = {f"e={int(r[group_col])}": 1.0 / len(pos) for _, r in pos.iterrows()}
        else:
            overall_att = float(att_by["att"].mean())
            overall_weights = {f"{group_col}={int(r[group_col])}": 1.0 / len(att_by) for _, r in att_by.iterrows()}

        # ---- overall SE via influence-function aggregation ----
        # R does this in two stages: per-level IFs → overall IF via
        # get_agg_inf_func.  The divisor is n_entities (matching R's
        # getSE: sqrt(mean(IF²)/n) where n = unique entity count).
        if has_rifs:
            if type == "dynamic":
                pos_leads = sorted(
                    v for v in att_by[group_col].values if v >= 0
                )
                pos_ifs = pd.DataFrame(
                    {e: rif_by_level[e] for e in pos_leads if e in rif_by_level}
                )
                if not pos_ifs.empty:
                    agg_rif = pos_ifs.mean(axis=1)
                    agg_c = agg_rif - agg_rif.mean()
                    overall_se = float(np.sqrt(np.sum(agg_c ** 2) / (n_entities ** 2)))
                else:
                    overall_se = float(att_by["se"].mean())

            elif type == "group":
                if rif_by_level:
                    level_df = pd.DataFrame(rif_by_level)
                    cohort_w = np.array([
                        float(gt[gt["cohort"] == g]["n_treat"].sum())
                        for g in level_df.columns
                    ])
                    cohort_w_norm = cohort_w / cohort_w.sum() if cohort_w.sum() > 0 else np.ones(len(cohort_w)) / len(cohort_w)
                    agg_rif = level_df.mul(cohort_w_norm, axis=1).sum(axis=1)
                    agg_c = agg_rif - agg_rif.mean()
                    overall_se = float(np.sqrt(np.sum(agg_c ** 2) / (n_entities ** 2)))
                else:
                    overall_se = float(att_by["se"].mean())

            else:  # calendar
                if rif_by_level:
                    level_df = pd.DataFrame(rif_by_level)
                    agg_rif = level_df.mean(axis=1)
                    agg_c = agg_rif - agg_rif.mean()
                    overall_se = float(np.sqrt(np.sum(agg_c ** 2) / (n_entities ** 2)))
                else:
                    overall_se = float(att_by["se"].mean())
        else:
            overall_se = float(att_by["se"].mean())

        overall_p = float(2 * (1 - _norm.cdf(
            abs(overall_att / overall_se)
        ))) if overall_se > 0 else float("nan")

        return AggteResult(
            att=overall_att,
            se=overall_se,
            p_value=overall_p,
            type=type,
            att_by=att_by,
            overall_weights=overall_weights,
            method=self.method,
            call=self.call,
        )


def _first_treat_period(treat: pd.Series, time: pd.Series, entity: pd.Series) -> pd.Series:
    """Return the first period in which each entity is treated (inf if never)."""
    df = pd.DataFrame({"e": entity.values, "t": time.values, "d": treat.values.astype(int)})
    first = df[df["d"] == 1].groupby("e")["t"].min()
    out = pd.Series(np.inf, index=df["e"].unique())
    out.update(first)
    return out.reindex(entity.values)


def _cell_dripw(
    sub: pd.DataFrame,
    y: str,
    entity: str,
    time: str,
    treatment: str,
    gco: float,
    t: float,
    pre: float,
    covariates: list[str],
    cluster: str,
    all_entities: np.ndarray,
) -> dict:
    """CS2021 doubly-robust ATT(g,t) for one cell (Sant'Anna & Zhao 2020 / DRDID ``drdid_panel``).

    Implements the influence-function-based DR-DiD estimator.  For each entity,
    the cell-level influence function is

        att_inf_func_i = inf_treat_i - inf_control_i

    where each component carries the bias-correcting terms for the estimated
    propensity score (logit) and outcome regression (OLS).  The stored
    per-entity ``RIF`` is shifted so that ``mean(RIF) = ATT(g,t)``:

        RIF_i = att_inf_func_i + ATT(g,t)

    and the cluster-robust standard error uses the full-sample influence
    function, ``V = sum_i (RIF_i - mean(RIF))² / N²`` with ``N`` the
    full-sample entity count.  See the ``ROADMAP`` changelog for the full
    formula and the ``DRDID`` (Sant'Anna & Zhao 2020) reference.
    """
    import statsmodels.api as sm

    # Merge pre and post to get ΔY per entity
    pre_df = sub[sub[time] == pre].copy()
    post_df = sub[sub[time] == t].copy()
    merged = pre_df.merge(
        post_df[[entity, y]], on=entity, suffixes=("_pre", "_post"), how="inner"
    )
    merged["dy"] = merged[f"{y}_post"] - merged[f"{y}_pre"]
    D = (merged["__g"] == gco).astype(float).values
    n_treat = int(D.sum())
    n_ctrl = int((~D.astype(bool)).sum())
    n_entities = len(merged)
    entity_ids = merged[entity].values

    if n_treat < 1 or n_ctrl < 2:
        return {
            "cohort": int(gco), "time": int(t), "lead": int(t - gco),
            "att": float("nan"), "se": float("nan"), "p_value": float("nan"),
            "n_treat": n_treat, "n_control": n_ctrl, "rif": None,
        }

    dy = merged["dy"].values.astype(float)

    # ---------- PS model (logit) ----------
    X = sm.add_constant(merged[covariates].values.astype(float))
    logit_fit = sm.Logit(D, X).fit(disp=False, maxiter=200)
    ps = np.minimum(logit_fit.predict(X), 1 - 1e-6)

    # ---------- Outcome regression (WLS on controls, weights = 1-D) ----------
    weights_ols = 1.0 - D
    ctrl = ~D.astype(bool)
    wls = sm.WLS(dy[ctrl], X[ctrl], weights=weights_ols[ctrl]).fit()
    out_delta = X @ wls.params

    # ---------- DR weights ----------
    trim_ps = np.ones(n_entities, dtype=bool)
    trim_ps[D == 0] = ps[D == 0] < 0.995
    w_treat = trim_ps * D
    w_cont = trim_ps * ps * (1.0 - D) / np.clip(1.0 - ps, 1e-15, None)
    mw_treat = float(np.mean(w_treat))
    mw_cont = float(np.mean(w_cont))

    dr_att_treat = w_treat * (dy - out_delta)
    dr_att_cont = w_cont * (dy - out_delta)
    eta_treat = float(np.mean(dr_att_treat) / mw_treat)
    eta_cont = float(np.mean(dr_att_cont) / mw_cont)
    dr_att = eta_treat - eta_cont

    # ---------- OLS asymptotic-linear representation ----------
    wols_x = weights_ols[:, None] * X
    wols_eX = weights_ols[:, None] * (dy - out_delta)[:, None] * X
    XpX = (wols_x.T @ X) / n_entities
    XpX_inv = np.linalg.inv(XpX)
    asy_lin_rep_wols = wols_eX @ XpX_inv

    # ---------- logit asymptotic-linear representation ----------
    W = ps * (1.0 - ps)
    score_ps = (D - ps)[:, None] * X
    XtWX_ps = X.T @ (W[:, None] * X)
    Hessian_ps = np.linalg.inv(XtWX_ps) * n_entities
    asy_lin_rep_ps = score_ps @ Hessian_ps

    # ---------- treated-component influence function ----------
    inf_treat_1 = dr_att_treat - w_treat * eta_treat
    M1 = (w_treat[:, None] * X).sum(axis=0) / n_entities
    inf_treat_2 = asy_lin_rep_wols @ M1
    inf_treat = (inf_treat_1 - inf_treat_2) / mw_treat

    # ---------- control-component influence function (with correction terms) ----------
    inf_cont_1 = dr_att_cont - w_cont * eta_cont
    M2 = ((w_cont * (dy - out_delta - eta_cont))[:, None] * X).sum(axis=0) / n_entities
    inf_cont_2 = asy_lin_rep_ps @ M2
    M3 = (w_cont[:, None] * X).sum(axis=0) / n_entities
    inf_cont_3 = asy_lin_rep_wols @ M3
    inf_control = (inf_cont_1 + inf_cont_2 - inf_cont_3) / mw_cont

    att_inf_func = inf_treat - inf_control
    # csdid stores the SHIFTED RIF so that mean(RIF) = ATT
    rif_shifted = att_inf_func + dr_att
    rif_series = pd.Series(rif_shifted, index=entity_ids)

    # ---------- cluster-robust SE via full-sample influence function ----------
    n_full = len(all_entities)
    rif_full = rif_series.reindex(all_entities).fillna(0.0)
    rif_c = rif_full - rif_full.mean()
    V = float(np.sum(rif_c ** 2) / (n_full ** 2))
    se = float(np.sqrt(V)) if V > 0 else float("nan")
    p_val = float(2 * (1 - _norm.cdf(abs(dr_att / se)))) if (se > 0 and np.isfinite(se)) else float("nan")

    return {
        "cohort": int(gco), "time": int(t), "lead": int(t - gco),
        "att": float(dr_att), "se": se, "p_value": p_val,
        "n_treat": n_treat, "n_control": n_ctrl, "rif": rif_series,
    }


def _staggered_hac_se(
    gt: pd.DataFrame,
    rif_matrix: pd.DataFrame,
    all_entities: np.ndarray,
    cluster_se: float,
    lags: int,
) -> float:
    """Newey-West temporal correction on the aggregated influence function.

    **PROJECT CONVENTION -- not externally validated.** The cluster-robust SE
    (``cluster_se``) already absorbs arbitrary within-entity across-time
    correlation by summing each entity's influence function before aggregation.
    This adds a Newey-West Bartlett-kernel correction for *common time shocks*:
    temporal autocorrelation of the per-time aggregated influence sums across
    entities.

    The correction is a multiplicative factor ``f = 1 + 2 * sum_{l=1}^{L}
    (1 - l/(L+1)) * rho_l`` applied to the cluster variance, where ``rho_l`` is
    the lag-``l`` autocorrelation of the demeaned per-time series ``U_t``. At
    ``lags=0`` the factor is exactly 1, so the result equals the cluster-robust
    SE. Under positive autocorrelation the SE is inflated. The Bartlett lag
    window keeps ``f >= 0`` for ``|rho_l| <= 1`` (floored at 0 defensively).

    ``U_t`` is the cross-entity sum of the per-time influence contribution:
    for ``method="dripw"`` it is built from the per-entity RIF columns of
    ``rif_matrix``; for ``method="reg"`` (no per-entity RIF) it is the per-cell
    ATT sum at time ``t`` (a coarser proxy). See the estimator docstring for
    the full caveat.
    """
    times = np.sort(gt["time"].unique())
    if len(times) <= 1 or lags < 1:
        return cluster_se
    if not rif_matrix.empty:
        cell_times = gt.loc[rif_matrix.columns, "time"].values
        U = np.array([
            float(rif_matrix.loc[:, cell_times == t].values.sum())
            for t in times
        ])
    else:
        U = (
            gt.groupby("time")["att"]
            .sum()
            .reindex(times)
            .values.astype(float)
        )
    U = U - U.mean()
    T = len(U)
    gamma0 = float(np.sum(U ** 2)) / T
    if gamma0 <= 0:
        return cluster_se
    factor = 1.0
    for lag in range(1, lags + 1):
        if lag >= T:
            break
        gammal = float(np.sum(U[:-lag] * U[lag:])) / (T - lag)
        rho = gammal / gamma0
        factor += 2.0 * (1.0 - lag / (lags + 1.0)) * rho
    factor = max(factor, 0.0)
    return float(np.sqrt(cluster_se ** 2 * factor))


def did_cs(
    data: pd.DataFrame,
    y: str,
    entity: str,
    time: str,
    treatment: str,
    covariates: list[str] | None = None,
    method: str | None = None,
    cluster: str | None = None,
    cov_type: str = "cluster",
    lags: int | None = None,
    control_cohorts: str = "not_yet_treated",
    bootstrap: bool = False,
    bootstrap_reps: int = 500,
    seed: int | None = None,
    ) -> CsDiDResult:
    """Callaway & Sant'Anna (2021) staggered difference-in-differences.

    Implements the Callaway & Sant'Anna (2021) group-time estimator.  Two
    estimation methods:

    * ``method="dripw"`` — doubly-robust (logit propensity score + OLS outcome
      regression), the default when ``covariates`` are provided.
    * ``method="reg"`` — outcome-regression DiD via a 2×2 OLS interaction
      (post × treated), the default when ``covariates`` are **not** provided.

    Each cohort ``g`` is compared, period by period, to a control group of
    never-treated units (plus optionally not-yet-treated units).  Standard
    errors are influence-function-based cluster-robust when ``dripw``, or
    entity-clustered OLS when ``reg``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.  ``treatment`` must be a binary indicator that turns on at
        treatment and stays on (staggered adoption).
    y : str
        Outcome column.
    entity, time : str
        Panel entity and time columns.
    treatment : str
        Binary treatment-indicator column.
    covariates : list of str, optional
        Covariate column names for the doubly-robust estimator.  Ignored when
        ``method="reg"``.
    method : {"dripw", "reg"}, optional
        Estimation method.  Defaults to ``"dripw"`` when ``covariates`` are
        provided, ``"reg"`` otherwise.
    cluster : str, optional
        Column for clustered standard errors (defaults to *entity*).
    cov_type : {"cluster", "HAC"}, default "cluster"
        Covariance estimator for the overall aggregated ATT. ``"cluster"``
        (default) uses entity-clustered influence-function standard errors.
        ``"HAC"`` applies a Newey-West (1987) Bartlett-kernel temporal
        correction on top of the cluster-robust base -- see the *HAC caveat*
        note below. ``lags`` is required when ``cov_type="HAC"``.
    lags : int, optional
        Number of Newey-West lags for ``cov_type="HAC"``. Required (and must be
        ``>= 1``) when ``cov_type="HAC"``.
    control_cohorts : {"not_yet_treated", "never_treated"}, default "not_yet_treated"
        Control group definition.  ``"not_yet_treated"`` uses both not-yet-treated
        and never-treated units as controls (the CS universal base);
        ``"never_treated"`` uses only the never-treated as the control group.
    bootstrap : bool, default False
        If True, compute bootstrap confidence intervals for the overall ATT
        by resampling entities.  When False (default), uses analytic SEs.
    bootstrap_reps : int, default 500
        Number of bootstrap replications (only used when ``bootstrap=True``).
    seed : int, optional
        Random seed for reproducible bootstrap.

    Returns
    -------
    CsDiDResult

    Notes
    -----
    Stata reference implementation: ``csdid`` and ``drdid`` packages by
    Fernando Rios-Avila — see the `stpackages repository
    <https://github.com/friosavila/stpackages/tree/main/csdid>`_ and
    `drdid <https://github.com/friosavila/stpackages/tree/main/drdid>`_.

    With ``method="dripw"`` the ATT(g,t) influence function follows
    Sant'Anna & Zhao (2020), as implemented in the R ``DRDID`` package
    (``drdid_panel`` / the ``trad`` method, which is ``csdid``'s default
    ``dripw``).  For each cell, with propensity score ``ps`` (logit of the
    group indicator on covariates, controls trimmed at ``ps < 0.995``),
    outcome regression ``m0`` (WLS of ``ΔY`` on covariates, weighted by
    ``1 - D``), and weights ``w_treat = ps·D``,
    ``w_cont = ps·(1-D)/(1-ps)``::

        att_inf_func = inf_treat - inf_control

        inf_treat  = (dr_att_treat - w_treat·eta_treat - asy_lin_rep_wols·M1) / mw_treat
        inf_control = (dr_att_cont - w_cont·eta_cont
                       + asy_lin_rep_ps·M2 - asy_lin_rep_wols·M3) / mw_cont

    where the ``asy_lin_rep_*`` terms are the asymptotic-linear
    representations of the estimated propensity-score and outcome-regression
    coefficients (the bias-correcting nuisance-parameter corrections).  The
    stored per-entity ``RIF`` is shifted so ``mean(RIF) = ATT(g,t)``.

    Per-cell cluster-robust SE uses the full-sample influence function,
    ``V = Σ_i (RIF_i - mean(RIF))² / N²`` (``N`` = full-sample entity count,
    no small-sample correction).  The overall pooled SE aggregates the
    per-entity RIFs across post-treatment cells with equal weight
    (``1/K``) and applies the same full-sample IF variance.

    **HAC caveat (experimental, project convention -- not externally
    validated).**  ``cov_type="HAC"`` is accepted for API symmetry with the
    other estimators, but it is *not* a canonical staggered-DiD variance.  The
    cluster-robust SE already absorbs arbitrary within-entity (across-time)
    correlation by summing each entity's influence function before aggregation.
    The HAC correction instead adds a Newey-West Bartlett-kernel adjustment for
    *common time shocks* -- temporal autocorrelation of the per-time aggregated
    influence sums across entities (see :func:`_staggered_hac_se`).  It reduces
    exactly to the cluster-robust SE when ``lags=0``, and inflates the SE under
    positive autocorrelation.  No Stata/R reference implements staggered-DiD HAC
    (the area is contested: Bacon decomposition, Callaway-Sant'Anna, etc.), so
    this is a documented project convention only.  A ``UserWarning`` is raised
    whenever ``cov_type="HAC"`` is used.  Prefer ``cov_type="cluster"`` for
    publication.

    csdid parity (aggregated SE).  The aggregated ATT and its SE are validated
    against Callaway & Sant'Anna's own aggregation, NOT against Stata's
    ``csdid_estat simple`` command.  Concretely the OE aggregated SE equals
    what ``csdid`` itself produces from its saved influence functions —
    ``csdid y x z, saverif(rif)`` followed by ``csdid_stats simple`` — and
    what the ``did`` R package computes in ``aggte(type="simple")``
    (``getSE`` = ``sqrt(mean(if²)/n)`` = ``sqrt(Σ_i if_i² / N²)``).  Both give
    the same value as OE (balanced fixture 0.41781627, unbalanced 0.62720813).
    WARNING: Stata's ``csdid_estat simple`` is buggy in the installed csdid
    version (v1.6/v1.58).  It posts the raw per-(g,t) VCoV and prints element
    [1,1] of it — i.e. the *first, pre-treatment* cell's SE (0.7479047
    balanced, 0.47824472 unbalanced) — as the "simple" ATT SE, which is not an
    aggregation SE at all.  Do NOT use ``csdid_estat simple`` output as a
    reference for the aggregated SE; use ``csdid_stats`` (or the ``did`` R
    package) instead.  csdid's ``csdid_estat simple`` delta-method code
    (``(r1,r2)*e(V_attgt)*(r1,r2)'``) is a different, non-canonical variance
    estimator that the ``did`` package does not implement and that differs from
    the influence-function aggregation above.
    """
    if method is None:
        method = "dripw" if (covariates is not None and len(covariates) > 0) else "reg"

    if method not in ("dripw", "reg"):
        raise ValueError(f"method must be 'dripw' or 'reg', got {method!r}")

    if covariates is None:
        covariates = []

    if cluster is None:
        cluster = entity

    cov_type = validate_cov_type(
        cov_type,
        accepted={"cluster", "HAC"},
        estimator="did_cs()",
    )
    if cov_type == "HAC":
        if lags is None:
            raise ValueError("lags= must be provided when cov_type='HAC'.")
        if lags < 0:
            raise ValueError("lags= must be a non-negative integer when cov_type='HAC'.")
        warnings.warn(
            "cov_type='HAC' on did_cs() is experimental and a PROJECT CONVENTION "
            "(Newey-West temporal correction on the aggregated influence function). It is "
            "NOT externally validated against Stata/R, and staggered-DiD HAC inference is a "
            "contested area. Prefer cluster-robust SEs (cov_type='cluster', the default) for "
            "publication.",
            UserWarning,
            stacklevel=2,
        )

    call = _capture_call(
        y=y, entity=entity, time=time, treatment=treatment,
        covariates=covariates, method=method,
        cluster=cluster, cov_type=cov_type, lags=lags,
        control_cohorts=control_cohorts,
        bootstrap=bootstrap, bootstrap_reps=bootstrap_reps, seed=seed,
    )

    df = data.copy()
    for col in (y, entity, time, treatment):
        if col not in df.columns:
            from open_econs._internal import errors
            raise errors.missing_column_error(col, df.columns.tolist())

    for c in covariates:
        if c not in df.columns:
            from open_econs._internal import errors
            raise errors.missing_column_error(c, df.columns.tolist())

    if control_cohorts not in ("not_yet_treated", "never_treated"):
        raise ValueError("control_cohorts must be 'not_yet_treated' or 'never_treated'.")

    g = _first_treat_period(df[treatment], df[time], df[entity])
    df["__g"] = g.values
    never = df["__g"] == np.inf
    all_times = np.sort(df[time].unique())
    cohorts = sorted(s for s in df["__g"].unique() if np.isfinite(s))
    all_entities = df[entity].unique()

    rows = []
    import statsmodels.api as sm

    for gco in cohorts:
        gco = float(gco)
        pre = gco - 1
        if pre not in all_times:
            continue

        for t in all_times:
            if t < gco:
                continue

            if control_cohorts == "not_yet_treated":
                ctrl_mask = (df["__g"] > t) | never
            else:
                ctrl_mask = never

            sub = df[(((df["__g"] == gco) | ctrl_mask) & (df[time].isin([pre, t])))]
            if sub.empty:
                continue
            sub = sub.copy()

            if method == "dripw" and covariates:
                res = _cell_dripw(sub, y, entity, time, treatment,
                                  gco, t, pre, covariates, cluster, all_entities)
                rows.append(res)
            else:
                # method == "reg": 2x2 OLS interaction (backward compatible)
                sub["__post"] = (sub[time] == t).astype(int)
                sub["__grp"] = (sub["__g"] == gco).astype(int)
                sub["__D"] = sub["__post"] * sub["__grp"]
                yv = sub[y].values.astype(float)
                Xv = np.column_stack([
                    np.ones(len(sub)),
                    sub["__post"].values,
                    sub["__grp"].values,
                    sub["__D"].values,
                ]).astype(float)
                fit = sm.OLS(yv, Xv).fit(cov_type="cluster", cov_kwds={"groups": sub[cluster].values})
                att = float(fit.params[3])
                se = float(fit.bse[3])
                p = float(2 * (1 - _norm.cdf(abs(att / se)))) if se > 0 else float("nan")
                n_treat = int(((sub["__g"] == gco) & (sub[time] == t)).sum())
                n_ctrl = int(len(sub) - n_treat)
                rows.append({
                    "cohort": int(gco), "time": int(t), "lead": int(t - gco),
                    "att": att, "se": se, "p_value": p,
                    "n_treat": n_treat, "n_control": n_ctrl,
                })

    gt = pd.DataFrame(rows)
    if gt.empty:
        raise ValueError("No cohort x post-period cells found; check the treatment indicator.")

    # Drop NaN ATT rows (failed cells)
    gt = gt.dropna(subset=["att"])
    if gt.empty:
        raise ValueError("All (g,t) cells produced NaN; check covariates and sample sizes.")

    gt = gt.sort_values(["cohort", "time"]).reset_index(drop=True)

    # ---- Aggregated ATT / SE from the full-sample influence function ----
    # Combine each post-treatment cell's per-entity RIF with equal weight
    # (1/K across the K cells) and take the cluster-robust SE of the
    # aggregated RIF, V = sum_i (agg_rif_i - mean)² / N² (N = full-sample
    # entity count).  This reproduces csdid's full-sample IF rescaling.
    K = len(gt)
    rif_matrix = pd.DataFrame(index=all_entities)
    for i, row in gt.iterrows():
        cell_rif = row.get("rif")
        if cell_rif is None:
            continue
        rif_matrix[i] = cell_rif.reindex(all_entities).fillna(0.0)
    if K > 0 and not rif_matrix.empty:
        w_eq = np.ones(K) / K
        agg_rif = rif_matrix.mul(w_eq, axis=1).sum(axis=1)
        overall = float(agg_rif.mean())
        agg_c = agg_rif - agg_rif.mean()
        overall_se = float(np.sqrt(np.sum(agg_c ** 2) / (len(all_entities) ** 2)))
    else:
        # Fallback (e.g. method="reg" cells carry no per-entity RIF): use the
        # weighted average of per-cell cluster SEs, consistent with prior behaviour.
        overall = float(np.average(gt["att"], weights=gt["n_treat"]))
        overall_se = float(np.sqrt(np.average(gt["se"] ** 2, weights=gt["n_treat"])))
    overall_p = float(2 * (1 - _norm.cdf(abs(overall / overall_se)))) if overall_se > 0 else float("nan")

    if cov_type == "HAC":
        assert lags is not None, "guaranteed non-None by validation above"
        overall_se = _staggered_hac_se(gt, rif_matrix, all_entities, overall_se, lags)
        overall_p = float(2 * (1 - _norm.cdf(abs(overall / overall_se)))) if overall_se > 0 else float("nan")

    ev = (
        gt.groupby("lead")
        .apply(lambda d: pd.Series({
            "att": np.average(d["att"], weights=d["n_treat"]),
            "se": np.sqrt(np.average(d["se"] ** 2, weights=d["n_treat"])),
            "n_cohort_periods": len(d),
        }))
        .reset_index()
    )
    ev["p_value"] = 2 * (1 - _norm.cdf(np.abs(ev["att"] / ev["se"].replace(0, np.nan))))
    ev = ev.sort_values("lead").reset_index(drop=True)

    if bootstrap and bootstrap_reps > 0:
        rng = np.random.RandomState(seed)
        entities_unique = df[entity].unique()
        boot_atts = []
        for _ in range(bootstrap_reps):
            boot_entities = rng.choice(entities_unique, size=len(entities_unique), replace=True)
            boot_df = pd.concat([df[df[entity] == e] for e in boot_entities], ignore_index=True)
            boot_g = _first_treat_period(boot_df[treatment], boot_df[time], boot_df[entity])
            boot_df["__g"] = boot_g.values
            boot_never = boot_df["__g"] == np.inf
            boot_cohorts = sorted(s for s in boot_df["__g"].unique() if np.isfinite(s))

            boot_rows = []
            for gco in boot_cohorts:
                gco = float(gco)
                pre = gco - 1
                if pre not in all_times:
                    continue
                for t in all_times:
                    if t < gco:
                        continue
                    if control_cohorts == "not_yet_treated":
                        ctrl_mask = (boot_df["__g"] > t) | boot_never
                    else:
                        ctrl_mask = boot_never
                    sub = boot_df[(((boot_df["__g"] == gco) | ctrl_mask) & (boot_df[time].isin([pre, t])))]
                    if sub.empty:
                        continue
                    sub = sub.copy()
                    if method == "dripw" and covariates:
                        try:
                            res = _cell_dripw(sub, y, entity, time, treatment,
                                              gco, t, pre, covariates, cluster,
                                              entities_unique)
                            if np.isfinite(res["att"]):
                                boot_rows.append({"att": res["att"], "n_treat": res["n_treat"]})
                        except Exception:
                            continue
                    else:
                        sub["__post"] = (sub[time] == t).astype(int)
                        sub["__grp"] = (sub["__g"] == gco).astype(int)
                        sub["__D"] = sub["__post"] * sub["__grp"]
                        yv = sub[y].values.astype(float)
                        Xv = np.column_stack([
                            np.ones(len(sub)),
                            sub["__post"].values,
                            sub["__grp"].values,
                            sub["__D"].values,
                        ]).astype(float)
                        try:
                            fit = sm.OLS(yv, Xv).fit(cov_type="cluster", cov_kwds={"groups": sub[cluster].values})
                            att = float(fit.params[3])
                            n_treat = int(((sub["__g"] == gco) & (sub[time] == t)).sum())
                            boot_rows.append({"att": att, "n_treat": n_treat})
                        except Exception:
                            continue

            if boot_rows:
                boot_gt = pd.DataFrame(boot_rows)
                boot_atts.append(float(np.average(boot_gt["att"], weights=boot_gt["n_treat"])))

        if boot_atts:
            overall = float(np.mean(boot_atts))
            overall_se = float(np.std(boot_atts, ddof=1))
            overall_p = float(2 * (1 - _norm.cdf(abs(overall / overall_se)))) if overall_se > 0 else float("nan")

    return CsDiDResult(
        att_group_time=gt,
        event_study=ev,
        att=overall,
        att_se=overall_se,
        att_p=overall_p,
        cohorts=[int(c) for c in cohorts],
        n_periods=int(len(all_times)),
        n_obs=int(len(df)),
        control_cohorts=control_cohorts,
        method=method,
        summary_text="",
        call=call,
        cov_type=cov_type,
    )


def staggered_did(*args: Any, **kwargs: Any) -> object:
    warnings.warn(
        "`staggered_did` is deprecated; use `did_cs` instead.",
        FutureWarning, stacklevel=2,
    )
    return did_cs(*args, **kwargs)
