"""Spike: compare open-econs estimators vs pyfixest (0.60.0).

Goal: judge whether pyfixest can sit *underneath* open-econs' wrappers
(keeping open-econs' public API: oe.ols / oe.fe / oe.iv with entity=/time=
kwargs and HC2 defaults) while improving correctness/precision.

Priority = correctness. So we check:
  (1) point estimates match to high precision (both do the same math),
  (2) SEs match for *equivalent* methods, and
  (3) flag where methods legitimately differ (and note which is more correct).

Run: python spikes/pyfixest_check.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyfixest as pf

import open_econs as oe


def _diff(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    idx = a.index.intersection(b.index)
    a = a.loc[idx]
    b = b.loc[idx]
    absd = (a - b).abs()
    denom = a.abs().where(a.abs() > 0, 1.0)
    reld = (absd / denom)
    return pd.DataFrame(
        {
            "oe": a.values,
            "pf": b.values,
            "abs_diff": absd.values,
            "rel_diff": reld.values,
        },
        index=idx,
    )


def report(name: str, oe_res, pf_coef, pf_se, note: str = ""):
    print("=" * 78)
    print(f"CASE: {name}")
    if note:
        print(f"  note: {note}")
    cmp = _diff(oe_res.coefficients, pf_coef)
    print("  -- coefficients --")
    print(cmp.to_string(float_format=lambda x: f"{x:.3e}"))
    if len(cmp):
        print(
            f"  coef max|diff|={cmp['abs_diff'].max():.3e}  "
            f"max rel diff={cmp['rel_diff'].max():.3e}"
        )
    cmp_se = _diff(oe_res.std_errors, pf_se)
    print("  -- std errors --")
    print(cmp_se.to_string(float_format=lambda x: f"{x:.3e}"))
    if len(cmp_se):
        print(
            f"  se max|diff|={cmp_se['abs_diff'].max():.3e}  "
            f"max rel diff={cmp_se['rel_diff'].max():.3e}"
        )
    print()


def main() -> None:
    d = pf.get_data().dropna().reset_index(drop=True)

    # --- Case A: OLS, no FE, HC2 (equivalent methods) ---
    oe_r = oe.ols("Y ~ X1 + X2", data=d, cov_type="HC2")
    pf_r = pf.feols("Y ~ X1 + X2", data=d, vcov="HC2")
    report("OLS no-FE HC2", oe_r, pf_r.coef(), pf_r.se(),
           note="methods equivalent: should match closely")

    # --- Case B: OLS, no FE, HC1 ---
    oe_r = oe.ols("Y ~ X1 + X2", data=d, cov_type="HC1")
    pf_r = pf.feols("Y ~ X1 + X2", data=d, vcov="HC1")
    report("OLS no-FE HC1", oe_r, pf_r.coef(), pf_r.se())

    # --- Case C: one-way FE, cluster CRV1 (equivalent-ish) ---
    oe_r = oe.fe("Y ~ X1", data=d, entity="f1", cluster="group_id")
    pf_r = pf.feols("Y ~ X1 | f1", data=d, vcov={"CRV1": "group_id"})
    report("1-way FE, CRV1 cluster(group_id)", oe_r, pf_r.coef(), pf_r.se(),
           note="pf applies fixest dof correction; oe uses statsmodels cluster")

    # --- Case D: two-way FE, cluster CRV1 ---
    oe_r = oe.fe("Y ~ X1", data=d, entity="f1", time="f2", cluster="group_id")
    pf_r = pf.feols("Y ~ X1 | f1 + f2", data=d, vcov={"CRV1": "group_id"})
    report("2-way FE, CRV1 cluster(group_id)", oe_r, pf_r.coef(), pf_r.se())

    # --- Case E: three-way FE (capability gap: oe.fe caps at 2) ---
    pf_r = pf.feols("Y ~ X1 | f1 + f2 + f3", data=d, vcov={"CRV1": "group_id"})
    print("=" * 78)
    print("CASE: 3-way FE (open-econs fe() CANNOT do this)")
    print("  pf.feols('Y ~ X1 | f1 + f2 + f3') coef:", pf_r.coef().to_dict())
    print("  pf std errors:", pf_r.se().to_dict())
    print()

    # --- Case F: multiway cluster, no FE (different estimators) ---
    oe_r = oe.ols("Y ~ X1 + X2", data=d, cluster=["f1", "f2"])
    pf_r = pf.feols("Y ~ X1 + X2", data=d, vcov={"CRV1": "f1+f2"})
    report("multiway cluster [f1,f2] no FE", oe_r, pf_r.coef(), pf_r.se(),
           note="DIFFERENT estimators: oe=minik (CGM2011), pf=multiway CRV1. "
                "coef must match; SEs will differ by design")

    # --- Case G: IV-2SLS ---
    try:
        oe_r = oe.iv("Y ~ 1 | X1 ~ Z1", data=d, cov_type="robust")
        pf_r = pf.feols("Y ~ 1 | X1 ~ Z1", data=d, vcov="iid")
        oe_c = oe_r.coefficients.rename({"exog": "Intercept", "endog": "X1"})
        oe_s = oe_r.std_errors.rename({"exog": "Intercept", "endog": "X1"})
        pf_c = pf_r.coef().rename(lambda x: str(x))
        pf_s = pf_r.se().rename(lambda x: str(x))
        print("=" * 78)
        print("CASE: IV-2SLS (X1 ~ Z1)")
        print("  note: oe uses linearmodels; pf uses fixest IV.")
        print("  coef oe:", oe_c.to_dict())
        print("  coef pf:", pf_c.to_dict())
        print("  coef max|diff|:", float((oe_c - pf_c).abs().max()))
        print("  (SE methods differ: linearmodels 'robust' vs fixest iid)")
        print()
    except Exception as e:
        print("IV case errored:", repr(e))

    # --- Case H: CRV3 cluster (open-econs has no CRV3) ---
    pf_r = pf.feols("Y ~ X1 + X2 | f1", data=d, vcov={"CRV3": "group_id"})
    print("=" * 78)
    print("CASE: CRV3 cluster(group_id) -- open-econs LACKS this (only minik/CRV1-ish)")
    print("  pf.feols CRV3 coef:", pf_r.coef().to_dict())
    print("  pf.feols CRV3 se  :", pf_r.se().to_dict())
    print()

    # --- Diagnostic: why do OLS HC2 SEs differ by ~0.15%? ---
    n, k = len(d), 3
    print("=" * 78)
    print("DIAGNOSTIC: OLS no-FE HC2 SE ratio oe/pf")
    oe_s = oe.ols("Y ~ X1 + X2", data=d, cov_type="HC2").std_errors
    pf_s = pf.feols("Y ~ X1 + X2", data=d, vcov="HC2").se()
    ratio = (oe_s / pf_s.rename(lambda x: str(x))).mean()
    print(f"  mean(oe_se / pf_se) = {ratio:.6f}")
    print(f"  sqrt(n/(n-k))       = {np.sqrt(n/(n-k)):.6f}  (pyfixest scales HC2 by this dof factor)")
    print()


if __name__ == "__main__":
    main()
