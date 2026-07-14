import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

import open_econs as oe
import statsmodels.api as sm


def _make_panel(n=50, T=6, seed=1):
    rng = np.random.default_rng(seed)
    ent = np.repeat(np.arange(n), T)
    t = np.tile(np.arange(T), n)
    year = 2000 + t
    x = rng.normal(size=n * T)
    z = rng.normal(size=n * T)
    g = rng.normal(size=n * T)
    y = 1.2 * x - 0.5 * z + g[ent] + rng.normal(size=n * T)
    return pd.DataFrame(
        {"y": y, "x": x, "z": z, "firm": ent, "year": year, "time": t}
    )


def test_multiway_one_way_matches_statsmodels():
    df = _make_panel()
    r = oe.ols("y ~ x + z", data=df, cluster="firm")
    X = sm.add_constant(df[["x", "z"]])
    smr = sm.OLS(df["y"], X).fit(cov_type="cluster", cov_kwds={"groups": df["firm"]})
    np.testing.assert_allclose(r.coefficients.values, smr.params.values, rtol=1e-9)
    np.testing.assert_allclose(r.std_errors.values, smr.bse.values, rtol=1e-6)


def _manual_minik(Xv, resid, groups):
    from itertools import combinations

    scores = Xv * resid[:, None]
    XtX_inv = np.linalg.inv(Xv.T @ Xv)

    def B(g):
        bb = np.zeros((Xv.shape[1], Xv.shape[1]))
        for gg in np.unique(g):
            s = scores[g == gg].sum(0)
            bb += np.outer(s, s)
        return bb

    Bt = np.zeros((Xv.shape[1], Xv.shape[1]))
    for size in range(1, len(groups) + 1):
        sign = -1 if size % 2 == 0 else 1
        for combo in combinations(range(len(groups)), size):
            inter = groups[combo[0]]
            for d in combo[1:]:
                inter = inter * (groups[d].max() + 2) + groups[d]
            Bt += sign * B(inter)
    V = XtX_inv @ Bt @ XtX_inv
    d = np.diag(V)
    d = np.where(d > 0, d, np.nan)
    return np.sqrt(d)


def test_multiway_two_way_matches_manual_minik():
    df = _make_panel()
    r = oe.ols("y ~ x + z", data=df, cluster=["firm", "year"])
    Xv = sm.add_constant(df[["x", "z"]]).values
    smr = sm.OLS(df["y"], Xv).fit()
    groups = [df["firm"].values.astype(int), df["year"].values.astype(int)]
    expected = _manual_minik(Xv, smr.resid.values, groups)
    np.testing.assert_allclose(r.std_errors.values, expected, rtol=1e-9, equal_nan=True)


def test_multiway_three_way_matches_independent_minik():
    df = _make_panel()
    df["industry"] = (df["firm"] + df["year"]) % 7
    r = oe.ols("y ~ x + z", data=df, cluster=["firm", "year", "industry"])

    from itertools import combinations

    X = sm.add_constant(df[["x", "z"]]).values
    smr = sm.OLS(df["y"], X).fit()
    scores = X * smr.resid.values[:, None]
    XtX_inv = np.linalg.inv(X.T @ X)
    groups = [
        df["firm"].values.astype(int),
        df["year"].values.astype(int),
        df["industry"].values.astype(int),
    ]

    def B(g):
        bb = np.zeros((3, 3))
        for gg in np.unique(g):
            s = scores[g == gg].sum(0)
            bb += np.outer(s, s)
        return bb

    Bt = np.zeros((3, 3))
    for size in range(1, 4):
        sign = -1 if size % 2 == 0 else 1
        for combo in combinations(range(3), size):
            inter = groups[combo[0]]
            for d in combo[1:]:
                inter = inter * (groups[d].max() + 2) + groups[d]
            Bt += sign * B(inter)
    Vexp = XtX_inv @ Bt @ XtX_inv
    Vexp_diag = np.where(np.diag(Vexp) > 0, np.diag(Vexp), np.nan)
    np.testing.assert_allclose(
        r.std_errors.values, np.sqrt(Vexp_diag), rtol=1e-9, equal_nan=True,
    )
    assert r.cov_type.startswith("cluster(")


def test_newey_west_matches_statsmodels():
    rng = np.random.default_rng(2)
    n = 200
    t = np.arange(n)
    x = rng.normal(size=n)
    ar_err = np.zeros(n)
    for i in range(1, n):
        ar_err[i] = 0.5 * ar_err[i - 1] + rng.normal()
    y = 0.8 * x + ar_err
    df = pd.DataFrame({"y": y, "x": x, "time": t})
    r = oe.ols("y ~ x", data=df, cov_type="HAC", lags=2, time="time")
    X = sm.add_constant(df[["x"]])
    smr = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    np.testing.assert_allclose(r.coefficients.values, smr.params.values, rtol=1e-9)
    np.testing.assert_allclose(r.std_errors.values, smr.bse.values, rtol=1e-6)


def test_newey_west_panel_cluster_finite():
    df = _make_panel()
    r = oe.ols("y ~ x + z", data=df, cov_type="HAC", lags=1, time="time", cluster="firm")
    assert np.all(np.isfinite(r.std_errors.values))
    assert "HAC(1)" in r.cov_type


def test_hac_adjust_lags0_matches_hc1():
    rng = np.random.default_rng(42)
    n = 100
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x": x, "time": np.arange(n)})
    r_hac = oe.ols("y ~ x", data=df, cov_type="HAC", lags=0, time="time", hac_adjust=True)
    r_hc1 = oe.ols("y ~ x", data=df, cov_type="HC1")
    np.testing.assert_allclose(r_hac.std_errors.values, r_hc1.std_errors.values, rtol=1e-10)


def test_hac_adjust_panel_cluster():
    df = _make_panel(n=10, T=12, seed=7)
    r0 = oe.ols(
        "y ~ x + z", data=df, cov_type="HAC", lags=1, time="time",
        cluster="firm", hac_adjust=False,
    )
    r1 = oe.ols(
        "y ~ x + z", data=df, cov_type="HAC", lags=1, time="time",
        cluster="firm", hac_adjust=True,
    )
    n_obs = len(df)
    k = 3  # intercept, x, z
    factor = np.sqrt(n_obs / (n_obs - k))
    np.testing.assert_allclose(r1.std_errors.values / r0.std_errors.values, factor, rtol=1e-10)
