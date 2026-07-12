import numpy as np
import numpy.testing as npt
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


@pytest.fixture
def df_did_panel() -> pd.DataFrame:
    np.random.seed(42)
    n_entities = 50
    n_periods = 10
    n = n_entities * n_periods
    treated = np.random.binomial(1, 0.5, n).astype(float)
    post = np.random.binomial(1, 0.5, n).astype(float)
    treat_post = treated * post
    y = 1.0 + 2.0 * treated + 3.0 * post + 5.0 * treat_post + np.random.normal(0, 2, n)
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)
    return pd.DataFrame({
        "y": y,
        "treated": treated,
        "post": post,
        "time": time,
        "entity": entity,
        "x1": np.random.normal(0, 1, n),
    })


class TestDiDHAC:
    def test_hac_se_differs_from_nonrobust(self, df_did_panel):
        r_hac = oe.did(
            "y ~ treated * post", data=df_did_panel,
            treatment="treated", post="post",
            cov_type="HAC", lags=2, time="time",
        )
        r_nonr = oe.did(
            "y ~ treated * post", data=df_did_panel,
            treatment="treated", post="post",
            cov_type="nonrobust",
        )
        assert not np.allclose(r_hac.std_errors, r_nonr.std_errors, rtol=1e-10)

    def test_hac_matches_manual_newway_west(self, df_did_panel):
        from open_econs.core.cov import newey_west_cov, _as_int_labels
        import statsmodels.api as sm
        from formulaic import Formula

        fml = Formula("y ~ treated * post")
        ms = fml.get_model_matrix(df_did_panel, na_action="drop")
        y = ms.lhs.values.ravel().astype(float)
        X = ms.rhs.values.astype(float)
        fitted = sm.OLS(y, X).fit(cov_type="nonrobust")
        time_labels = _as_int_labels(df_did_panel.loc[ms.rhs.index, "time"].values)
        V = newey_west_cov(X, fitted.resid, max_lags=2, cluster=time_labels, adjust=False)
        manual_se = np.sqrt(np.diag(V))

        r = oe.did(
            "y ~ treated * post", data=df_did_panel,
            treatment="treated", post="post",
            cov_type="HAC", lags=2, time="time",
        )
        npt.assert_allclose(r.std_errors.values, manual_se, rtol=1e-10, atol=1e-12)

    def test_hac_requires_lags(self, df_did_panel):
        with pytest.raises(ValueError, match="lags"):
            oe.did(
                "y ~ treated * post", data=df_did_panel,
                treatment="treated", post="post",
                cov_type="HAC", lags=None, time="time",
            )

    def test_hac_requires_time(self, df_did_panel):
        with pytest.raises(ValueError, match="time"):
            oe.did(
                "y ~ treated * post", data=df_did_panel,
                treatment="treated", post="post",
                cov_type="HAC", lags=2, time=None,
            )

    def test_hac_cov_label(self, df_did_panel):
        r = oe.did(
            "y ~ treated * post", data=df_did_panel,
            treatment="treated", post="post",
            cov_type="HAC", lags=2, time="time",
        )
        assert r.cov_type == "HAC(2)"

    def test_hac_alias_lowercase(self, df_did_panel):
        r1 = oe.did(
            "y ~ treated * post", data=df_did_panel,
            treatment="treated", post="post",
            cov_type="HAC", lags=2, time="time",
        )
        r2 = oe.did(
            "y ~ treated * post", data=df_did_panel,
            treatment="treated", post="post",
            cov_type="hac", lags=2, time="time",
        )
        npt.assert_allclose(r1.std_errors.values, r2.std_errors.values, rtol=1e-12)

    def test_hac_with_covariates(self, df_did_panel):
        r = oe.did(
            "y ~ treated * post + x1", data=df_did_panel,
            treatment="treated", post="post",
            cov_type="HAC", lags=2, time="time",
        )
        assert len(r.coefficients) == 5
        assert r.cov_type == "HAC(2)"

    def test_hac_adjust_changes_se(self, df_did_panel):
        r_adj = oe.did(
            "y ~ treated * post", data=df_did_panel,
            treatment="treated", post="post",
            cov_type="HAC", lags=2, time="time", hac_adjust=True,
        )
        r_no = oe.did(
            "y ~ treated * post", data=df_did_panel,
            treatment="treated", post="post",
            cov_type="HAC", lags=2, time="time", hac_adjust=False,
        )
        assert not np.allclose(r_adj.std_errors, r_no.std_errors, rtol=1e-10)

    def test_hac_preserves_did_coefficient(self, df_did_panel):
        r_hac = oe.did(
            "y ~ treated * post", data=df_did_panel,
            treatment="treated", post="post",
            cov_type="HAC", lags=2, time="time",
        )
        r_nonr = oe.did(
            "y ~ treated * post", data=df_did_panel,
            treatment="treated", post="post",
            cov_type="nonrobust",
        )
        npt.assert_allclose(r_hac.did_coefficient, r_nonr.did_coefficient, rtol=1e-12)