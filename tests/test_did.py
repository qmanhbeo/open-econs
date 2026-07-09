import numpy as np
import pandas as pd
import pytest

import open_econs as oe


@pytest.fixture
def df_did() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    treated = np.random.binomial(1, 0.5, n).astype(float)
    post = np.random.binomial(1, 0.5, n).astype(float)
    treat_post = treated * post
    y = 1.0 + 2.0 * treated + 3.0 * post + 5.0 * treat_post + np.random.normal(0, 2, n)
    return pd.DataFrame({
        "y": y,
        "treated": treated,
        "post": post,
        "x1": np.random.normal(0, 1, n),
    })


class TestDiD:
    def test_basic_did(self, df_did):
        r = oe.did("y ~ treated * post", data=df_did, treatment="treated", post="post")
        assert hasattr(r, "did_coefficient")
        assert hasattr(r, "did_std_error")

    def test_coefficient_count(self, df_did):
        r = oe.did("y ~ treated * post", data=df_did, treatment="treated", post="post")
        assert len(r.coefficients) == 4

    def test_tidy_shape(self, df_did):
        r = oe.did("y ~ treated * post", data=df_did, treatment="treated", post="post")
        tidy = r.tidy()
        expected = {"Variable", "Coef", "Std Err", "t", "P>|t|", "0.025", "0.975"}
        assert set(tidy.columns) == expected
        assert len(tidy) == 4

    def test_summary(self, df_did):
        r = oe.did("y ~ treated * post", data=df_did, treatment="treated", post="post")
        s = str(r)
        assert "Difference-in-Differences" in s

    def test_nobs(self, df_did):
        r = oe.did("y ~ treated * post", data=df_did, treatment="treated", post="post")
        assert r.nobs == len(df_did)

    def test_did_positive_effect(self, df_did):
        r = oe.did("y ~ treated * post", data=df_did, treatment="treated", post="post")
        assert r.did_coefficient > 0

    def test_clustering(self, df_did):
        df_did["cluster"] = np.random.choice(10, len(df_did))
        r = oe.did(
            "y ~ treated * post", data=df_did,
            treatment="treated", post="post", cluster="cluster",
        )
        assert r.cluster_var == "cluster"
        assert r.did_coefficient is not None

    def test_covariates(self, df_did):
        r = oe.did(
            "y ~ treated * post + x1", data=df_did,
            treatment="treated", post="post",
        )
        assert len(r.coefficients) == 5

    def test_missing_column_raises(self, df_did):
        with pytest.raises(ValueError, match="Column.*not found"):
            oe.did("y ~ nonexistent * post", data=df_did, treatment="nonexistent", post="post")

    def test_immutability(self, df_did):
        r = oe.did("y ~ treated * post", data=df_did, treatment="treated", post="post")
        with pytest.raises(AttributeError, match="immutable"):
            r.new_attr = 42

    def test_vcov(self, df_did):
        r = oe.did("y ~ treated * post", data=df_did, treatment="treated", post="post")
        v = r.vcov()
        assert isinstance(v, pd.DataFrame)
        assert v.shape == (4, 4)

    def test_export_json(self, df_did, tmp_path):
        r = oe.did("y ~ treated * post", data=df_did, treatment="treated", post="post")
        path = tmp_path / "did_result.json"
        r.export(str(path))
        assert path.exists()

    def test_context_did(self, df_did):
        ctx = oe.Context(df_did)
        r = ctx.did("y ~ treated * post", treatment="treated", post="post")
        assert r.did_coefficient > 0


class TestBalance:
    def test_basic_balance(self, df_did):
        b = oe.balance(df_did, treatment="treated")
        assert isinstance(b, pd.DataFrame)
        assert "t-statistic" in b.columns
        assert len(b) >= 1

    def test_balance_specific_covariates(self, df_did):
        b = oe.balance(df_did, treatment="treated", covariates=["x1"])
        assert len(b) == 1
        assert b.iloc[0]["Variable"] == "x1"

    def test_context_balance(self, df_did):
        ctx = oe.Context(df_did)
        b = ctx.balance(treatment="treated")
        assert isinstance(b, pd.DataFrame)

    def test_balance_invalid_treatment(self, df_did):
        with pytest.raises(ValueError, match="exactly 2"):
            oe.balance(df_did, treatment="x1")