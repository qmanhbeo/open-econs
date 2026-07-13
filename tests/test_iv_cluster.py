import numpy as np
import pandas as pd
import pytest

import open_econs as oe
from linearmodels.iv import IV2SLS


def _make_data(seed=0, n=200, n_firm=50):
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firm), n // n_firm)
    z = rng.standard_normal(n)
    w = rng.standard_normal(n)
    x = 0.5 * z + 0.3 * w + rng.standard_normal(n)
    y = 1.0 * x + 0.5 * w + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "x": x, "w": w, "z": z, "firm": firm})


def test_iv_cluster_matches_linearmodels():
    df = _make_data()
    r = oe.iv("y ~ w | x ~ z", data=df, cluster="firm")

    const = pd.Series(1.0, index=df.index, name="Intercept")
    f = IV2SLS(
        df["y"],
        pd.concat([const, df[["w"]]], axis=1),
        df[["x"]],
        df[["z"]],
    ).fit(cov_type="clustered", clusters=df["firm"].values)

    assert np.allclose(r.coefficients.values, f.params.values, rtol=1e-6)
    assert np.allclose(r.std_errors.values, f.std_errors.values, rtol=1e-6)
    assert np.allclose(r.z_stats.values, f.tstats.values, rtol=1e-6)
    assert r.cov_type == "clustered(firm)"


def test_iv_cluster_differs_from_robust():
    df = _make_data()
    r_clust = oe.iv("y ~ w | x ~ z", data=df, cluster="firm")
    r_robust = oe.iv("y ~ w | x ~ z", data=df, cov_type="robust")
    assert not np.allclose(r_clust.std_errors.values, r_robust.std_errors.values)


def test_iv_cluster_cov_type_without_cluster_raises():
    df = _make_data()
    with pytest.raises(ValueError, match="cluster"):
        oe.iv("y ~ w | x ~ z", data=df, cov_type="clustered")


def test_iv_multiway_cluster_not_implemented():
    df = _make_data()
    with pytest.raises(NotImplementedError, match="multi-way"):
        oe.iv("y ~ w | x ~ z", data=df, cluster=["firm", "w"])


def test_iv_cluster_missing_column_raises():
    df = _make_data()
    with pytest.raises(ValueError, match="not found"):
        oe.iv("y ~ w | x ~ z", data=df, cluster="nope")


def test_iv_cluster_label_in_summary():
    df = _make_data()
    r = oe.iv("y ~ w | x ~ z", data=df, cluster="firm")
    assert "clustered(firm)" in r.summary()
