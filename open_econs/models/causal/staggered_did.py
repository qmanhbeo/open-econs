from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from open_econs._version import __version__
from open_econs.core.base import BaseModel
from open_econs.core.call_capture import capture_call as _capture_call


class StaggeredDiDResult(BaseModel):
    """Result of a staggered difference-in-differences estimator.

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
) -> dict:
    """CS2021 doubly-robust ATT(g,t) for one cell.

    Computes the ATT via the influence-function moment condition:

        RIF_i = (D_i/p_D  -  w_i/mean_w) * (ΔY_i - m_0(X_i))
        ATT   = mean(RIF)

    where D = treatment indicator, p_D = mean(D),
    w_i = (1-D_i) * p(X_i)/(1-p(X_i)), m_0(X) = E[ΔY | X, D=0].
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

    if n_treat < 1 or n_ctrl < 2:
        return {
            "cohort": int(gco), "time": int(t), "lead": int(t - gco),
            "att": float("nan"), "se": float("nan"), "p_value": float("nan"),
            "n_treat": n_treat, "n_control": n_ctrl,
        }

    dy = merged["dy"].values.astype(float)
    cl_groups = merged[cluster].values if cluster else None

    # ---------- PS model (logit) ----------
    X = sm.add_constant(merged[covariates].values.astype(float))
    logit_fit = sm.Logit(D, X).fit(disp=False, maxiter=100)
    p = logit_fit.predict(X)

    # ---------- Outcome regression (OLS on controls at base period) ----------
    ctrl_idx = ~D.astype(bool)
    ols_fit = sm.OLS(dy[ctrl_idx], X[ctrl_idx]).fit()
    m0 = ols_fit.predict(X)

    # ---------- RIF ----------
    w = (1.0 - D) * p / np.clip(1.0 - p, 1e-15, None)
    p_D = float(np.mean(D))
    mean_w = float(np.mean(w))

    # Handle edge cases where PS is extreme
    if p_D <= 0 or mean_w <= 0:
        return {
            "cohort": int(gco), "time": int(t), "lead": int(t - gco),
            "att": float("nan"), "se": float("nan"), "p_value": float("nan"),
            "n_treat": n_treat, "n_control": n_ctrl,
        }

    rif = (D / p_D - w / mean_w) * (dy - m0)
    att = float(np.mean(rif))

    # ---------- Cluster-robust SE via aggregated influence functions ----------
    rif_c = rif - att
    if cluster and cl_groups is not None:
        cl_df = pd.DataFrame({"rif_c": rif_c, "cl": cl_groups})
        cl_agg = cl_df.groupby("cl")["rif_c"].sum()
        V = float((cl_agg ** 2).sum() / (n_entities ** 2))
    else:
        V = float(np.sum(rif_c ** 2) / (n_entities ** 2))

    se = float(np.sqrt(V)) if V > 0 else float("nan")
    p_val = float(2 * (1 - _norm.cdf(abs(att / se)))) if (se > 0 and np.isfinite(se)) else float("nan")

    return {
        "cohort": int(gco), "time": int(t), "lead": int(t - gco),
        "att": att, "se": se, "p_value": p_val,
        "n_treat": n_treat, "n_control": n_ctrl,
    }


def staggered_did(
    data: pd.DataFrame,
    y: str,
    entity: str,
    time: str,
    treatment: str,
    covariates: list[str] | None = None,
    method: str | None = None,
    cluster: str | None = None,
    control_cohorts: str = "not_yet_treated",
    bootstrap: bool = False,
    bootstrap_reps: int = 500,
    seed: int | None = None,
) -> StaggeredDiDResult:
    """Staggered (heterogeneous-timing) difference-in-differences.

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
    StaggeredDiDResult
    """
    if method is None:
        method = "dripw" if (covariates is not None and len(covariates) > 0) else "reg"

    if method not in ("dripw", "reg"):
        raise ValueError(f"method must be 'dripw' or 'reg', got {method!r}")

    if covariates is None:
        covariates = []

    if cluster is None:
        cluster = entity

    call = _capture_call(
        y=y, entity=entity, time=time, treatment=treatment,
        covariates=covariates, method=method,
        cluster=cluster, control_cohorts=control_cohorts,
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
                                  gco, t, pre, covariates, cluster)
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
    overall = float(np.average(gt["att"], weights=gt["n_treat"]))
    overall_se = float(np.sqrt(np.average(gt["se"] ** 2, weights=gt["n_treat"])))
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
                                              gco, t, pre, covariates, cluster)
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

    return StaggeredDiDResult(
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
    )
