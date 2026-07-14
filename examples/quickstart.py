"""Quick-start examples for open-econs.

Each example maps to a familiar Stata / R workflow so Stata users can migrate
easily.  Run with::

    python examples/quickstart.py

(Plots are skipped unless matplotlib is installed; all estimators are
headless and safe to run in CI.)
"""

import numpy as np
import pandas as pd

import open_econs as oe


def _panel(n=200, T=8, seed=0):
    rng = np.random.default_rng(seed)
    ent = np.repeat(np.arange(n), T)
    t = np.tile(np.arange(T), n)
    mu = rng.normal(0, 1, size=n)
    x = rng.normal(size=n * T)
    y = 0.5 * x + mu[ent] + rng.normal(size=n * T)
    return pd.DataFrame({"y": y, "x": x, "firm": ent, "year": t})


def main() -> None:
    df = _panel()

    # Stata: reg y x
    print(oe.ols("y ~ x", data=df).summary())

    # Stata: xtreg y x, fe  (entity + time fixed effects)
    pc = oe.PanelContext(df, entity="firm", time="year")
    print(pc.fe("y ~ x").summary())

    # Stata: xtreg y x, re
    print(pc.re("y ~ x").summary())

    # Stata: ivregress 2sls y (x = z)
    df["z"] = df["x"] + np.random.normal(size=len(df))
    print(oe.iv("y ~ x | z", data=df).summary())

    # Stata: xtabond / xtdpdsys style dynamic panel (Arellano-Bond)
    print(oe.abond("y ~ x", data=df, entity="firm", time="year").summary())

    # Stata: did / staggered DiD (Callaway & Sant'Anna)
    rng = np.random.default_rng(5)
    ad = rng.choice([2, 4, 6, 99], size=df["firm"].nunique())
    treat = np.zeros(len(df))
    for e in range(df["firm"].nunique()):
        m = df["firm"] == e
        treat[m.values] = (df.loc[m, "year"] >= ad[e]).astype(float).values
    df["treat"] = treat
    print(oe.did_cs(df, y="y", entity="firm", time="year", treatment="treat").summary())

    # Regression discontinuity
    rx = rng.uniform(-1, 1, size=1000)
    ry = rx + 2.0 * (rx >= 0).astype(float) + rng.normal(size=1000)
    rdd_df = pd.DataFrame({"y": ry, "x": rx, "treat": (rx >= 0).astype(float)})
    print(oe.rdd(rdd_df, y="y", running="x", cutoff=0.0).summary())


if __name__ == "__main__":
    main()
