from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from open_econs._version import __version__
from open_econs.core.base import BaseModel
from open_econs.core.call_capture import capture_call as _capture_call


class StaggeredDiDResult(BaseModel):
    """Result of a Callaway & Sant'Anna (2021) staggered DiD estimator.

    Stores group-time average treatment effects ``att_group_time`` (one row per
    cohort x period), the aggregated dynamic ``event_study`` path (lead l >= 0),
    and the overall pooled ``att``.  Inference uses entity-clustered standard
    errors from the underlying two-way regressions.
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
        self.call = call
        self.timestamp = __import__("datetime").datetime.now()
        self.package_version = __version__
        self._freeze()

    def tidy(self) -> pd.DataFrame:
        return self.att_group_time

    def summary(self) -> str:
        lines = [
            "        Callaway & Sant'Anna (2021) Staggered DiD Results        ",
            "==================================================================",
            f"  Overall ATT           : {self.att:.4f} (se {self.att_se:.4f}, p {self.att_p:.3f})",
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


def staggered_did(
    data: pd.DataFrame,
    y: str,
    entity: str,
    time: str,
    treatment: str,
    cluster: str | None = None,
    control_cohorts: str = "not_yet_treated",
) -> StaggeredDiDResult:
    """Staggered (heterogeneous-timing) difference-in-differences.

    Implements the Callaway & Sant'Anna (2021) group-time estimator.  Each
    cohort ``g`` (the period an entity is first treated) is compared, period by
    period, to a control group of units that are *not yet treated* by period
    ``t`` (plus the never-treated), in a regression with cohort and time fixed
    effects.  The coefficient on the cohort-specific treatment dummy is the
    group-time ATT(g, t).  These are aggregated into a dynamic event-study path
    and an overall pooled ATT.

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
    cluster : str, optional
        Column for clustered standard errors (defaults to *entity*).
    control_cohorts : {"not_yet_treated", "never_treated"}, default "not_yet_treated"
        Control group definition.  ``"not_yet_treated"`` uses both not-yet-treated
        and never-treated units as controls (the CS universal base);
        ``"never_treated"`` uses only the never-treated as the control group.

    Returns
    -------
    StaggeredDiDResult
    """
    if cluster is None:
        cluster = entity
    call = _capture_call(
        y=y, entity=entity, time=time, treatment=treatment,
        cluster=cluster, control_cohorts=control_cohorts,
    )
    df = data.copy()
    for col in (y, entity, time, treatment):
        if col not in df.columns:
            from open_econs._internal import errors

            raise errors.missing_column_error(col, df.columns.tolist())
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
            # No pre-treatment period available for this cohort.
            continue
        for t in all_times:
            if t < gco:
                continue
            # Clean control set for (g, t): units not yet treated by t, plus the
            # never-treated.  Used at both the pre (g-1) and post (t) periods.
            if control_cohorts == "not_yet_treated":
                ctrl_mask = (df["__g"] > t) | never
            else:
                ctrl_mask = never
            sub = df[(((df["__g"] == gco) | ctrl_mask) & (df[time].isin([pre, t])))]
            if sub.empty:
                continue
            sub = sub.copy()
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
            rows.append(
                {"cohort": int(gco), "time": int(t), "lead": int(t - gco),
                 "att": att, "se": se, "p_value": p, "n_treat": n_treat, "n_control": n_ctrl}
            )

    gt = pd.DataFrame(rows)
    if gt.empty:
        raise ValueError("No cohort x post-period cells found; check the treatment indicator.")

    # Aggregate: overall (cohort-size weighted) and event-study by lead.
    gt = gt.sort_values(["cohort", "time"]).reset_index(drop=True)
    overall = float(np.average(gt["att"], weights=gt["n_treat"]))
    # Overall SE via the weighted average of the per-cell SEs (conservative).
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
        summary_text="",
        call=call,
    )
