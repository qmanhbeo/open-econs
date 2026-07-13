"""Micro-benchmark: open_econs ``ols``/``fe`` vs statsmodels / linearmodels.

Generated panel: 2,000 entities x 5 time periods = 10,000 observations, 10
covariates, plus an entity fixed effect and Gaussian noise.  Reports real
wall-clock timings and the max-absolute coefficient delta (correctness) for
each engine.

Run:
    python benchmarks/ols_fe.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import open_econs as oe


def make_panel(n_entities: int = 2000, t_max: int = 5, k: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = n_entities * t_max
    entities = np.repeat(np.arange(n_entities), t_max)
    times = np.tile(np.arange(t_max), n_entities)

    X = rng.normal(size=(n, k))
    beta = rng.normal(size=k)
    entity_fe = rng.normal(size=n_entities)[entities]
    noise = rng.normal(scale=0.5, size=n)

    y = X @ beta + entity_fe + noise
    cols = {f"x{i+1}": X[:, i] for i in range(k)}
    df = pd.DataFrame({"y": y, "entity": entities, "time": times, **cols})
    return df, beta


def _time(fn, repeats: int = 3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return res, best


def main() -> None:
    df, beta = make_panel()
    k = 10
    xnames = [f"x{i+1}" for i in range(k)]
    formula = "y ~ " + " + ".join(xnames)

    print(f"Panel: {len(df):,} obs, {k} covariates, 2,000 entities x 5 time\n")

    # ── OLS: open_econs vs statsmodels ─────────────────────────────────────
    import statsmodels.api as sm

    (r_oe, oe_t) = _time(lambda: oe.ols(formula, df))
    (sm_fit, sm_t) = _time(
        lambda: sm.OLS(df["y"], sm.add_constant(df[xnames])).fit()
    )
    oe_coef = np.array([float(r_oe.coefficients[n]) for n in xnames])
    sm_coef = sm_fit.params[xnames].to_numpy(dtype=float)
    ols_delta = float(np.max(np.abs(oe_coef - sm_coef)))

    # ── FE (entity): open_econs vs linearmodels ─────────────────────────────
    from linearmodels.panel import PanelOLS

    (r_fe, oe_fe_t) = _time(lambda: oe.fe(formula, df, entity="entity"))
    pdf = df.set_index(["entity", "time"])
    (lm_fit, lm_t) = _time(
        lambda: PanelOLS(pdf["y"], pdf[xnames], entity_effects=True).fit()
    )
    fe_coef = np.array([float(r_fe.coefficients[n]) for n in xnames])
    lm_coef = lm_fit.params[xnames].to_numpy(dtype=float)
    fe_delta = float(np.max(np.abs(fe_coef - lm_coef)))

    # ── report ───────────────────────────────────────────────────────────────
    print(f"{'estimator':<10} {'open_econs (s)':>14} {'reference (s)':>14} "
          f"{'speedup':>9} {'max|dCoef|':>12}")
    print("-" * 63)
    print(f"{'ols':<10} {oe_t:>14.4f} {sm_t:>14.4f} "
          f"{sm_t / oe_t:>8.2f}x {ols_delta:>12.2e}")
    print(f"{'fe':<10} {oe_fe_t:>14.4f} {lm_t:>14.4f} "
          f"{lm_t / oe_fe_t:>8.2f}x {fe_delta:>12.2e}")
    print()
    print(f"OLS coefficient delta vs statsmodels : {ols_delta:.3e}")
    print(f"FE  coefficient delta vs linearmodels: {fe_delta:.3e}")


if __name__ == "__main__":
    main()
