import numpy as np
import pandas as pd
import pytest

import open_econs as oe


@pytest.fixture
def df_binary() -> pd.DataFrame:
    np.random.seed(42)
    n = 500
    x1 = np.random.uniform(-1, 1, n)
    x2 = np.random.uniform(-1, 1, n)
    log_odds = 0.5 + 1.2 * x1 - 0.8 * x2
    p = 1 / (1 + np.exp(-log_odds))
    y = (np.random.uniform(0, 1, n) < p).astype(float)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


class TestLogit:
    def test_basic_logit(self, df_binary):
        r = oe.logit("y ~ x1 + x2", data=df_binary)
        assert isinstance(r.coefficients, pd.Series)
        assert len(r.coefficients) == 3
        assert "Intercept" in r.coefficients.index
        assert r.model_type == "logit"

    def test_tidy_shape(self, df_binary):
        r = oe.logit("y ~ x1 + x2", data=df_binary)
        tidy = r.tidy()
        expected = {"Variable", "Coef", "Std Err", "z", "P>|z|", "0.025", "0.975"}
        assert set(tidy.columns) == expected
        assert len(tidy) == 3

    def test_summary_returns_string(self, df_binary):
        r = oe.logit("y ~ x1 + x2", data=df_binary)
        s = r.summary()
        assert isinstance(s, str)
        assert "LOGIT" in s

    def test_pseudo_r2_range(self, df_binary):
        r = oe.logit("y ~ x1 + x2", data=df_binary)
        assert 0 <= r.pseudo_r2 <= 1

    def test_margins_shape(self, df_binary):
        r = oe.logit("y ~ x1 + x2", data=df_binary)
        m = r.margins()
        assert isinstance(m, pd.DataFrame)
        assert "dy/dx" in m.columns
        assert len(m) == 2  # margins exclude intercept

    def test_predict_proba(self, df_binary):
        r = oe.logit("y ~ x1 + x2", data=df_binary)
        pred = r.predict(proba=True)
        assert len(pred) == len(df_binary)
        assert pred.name == "predicted_proba"
        assert pred.between(0, 1).all()

    def test_predict_class(self, df_binary):
        r = oe.logit("y ~ x1 + x2", data=df_binary)
        pred = r.predict(proba=False)
        assert set(np.unique(pred.values)).issubset({0, 1})

    def test_predict_newdata(self, df_binary):
        r = oe.logit("y ~ x1 + x2", data=df_binary)
        new = df_binary[["x1", "x2"]].head(10)
        pred = r.predict(newdata=new)
        assert len(pred) == 10

    def test_immutability(self, df_binary):
        r = oe.logit("y ~ x1 + x2", data=df_binary)
        with pytest.raises(AttributeError, match="immutable"):
            r.new_attr = 42

    def test_coefficients_match_statsmodels(self, df_binary):
        import statsmodels.api as sm
        y = df_binary["y"]
        X = sm.add_constant(df_binary[["x1", "x2"]])
        sm_r = sm.Logit(y, X).fit(disp=False)
        oe_r = oe.logit("y ~ x1 + x2", data=df_binary)
        np.testing.assert_allclose(oe_r.coefficients.values, sm_r.params, rtol=1e-8)
        np.testing.assert_allclose(oe_r.std_errors.values, sm_r.bse, rtol=1e-6)

    def test_llf_aic_bic(self, df_binary):
        r = oe.logit("y ~ x1 + x2", data=df_binary)
        assert not np.isnan(r.llf)
        assert not np.isnan(r.aic)
        assert not np.isnan(r.bic)

    def test_nobs_df(self, df_binary):
        r = oe.logit("y ~ x1 + x2", data=df_binary)
        assert r.nobs == len(df_binary)
        assert r.df_resid == r.nobs - r.df_model - 1

    def test_conf_int_structure(self, df_binary):
        r = oe.logit("y ~ x1 + x2", data=df_binary)
        assert list(r.conf_int.columns) == ["lower", "upper"]
        assert (r.conf_int["lower"] < r.conf_int["upper"]).all()

    def test_no_intercept(self, df_binary):
        r = oe.logit("y ~ x1 + x2 - 1", data=df_binary)
        assert "Intercept" not in r.coefficients.index

    def test_rhs_formula(self, df_binary):
        r = oe.logit("y ~ x1 + x2", data=df_binary)
        assert r.rhs_formula == "x1 + x2"

    def test_missing_column_raises(self, df_binary):
        with pytest.raises(ValueError, match="Column.*not found"):
            oe.logit("y ~ nonexistent", data=df_binary)


class TestProbit:
    def test_basic_probit(self, df_binary):
        r = oe.probit("y ~ x1 + x2", data=df_binary)
        assert isinstance(r.coefficients, pd.Series)
        assert r.model_type == "probit"

    def test_margins_shape(self, df_binary):
        r = oe.probit("y ~ x1 + x2", data=df_binary)
        m = r.margins()
        assert isinstance(m, pd.DataFrame)
        assert len(m) == 2  # margins exclude intercept

    def test_coefficients_match_statsmodels(self, df_binary):
        import statsmodels.api as sm
        y = df_binary["y"]
        X = sm.add_constant(df_binary[["x1", "x2"]])
        sm_r = sm.Probit(y, X).fit(disp=False)
        oe_r = oe.probit("y ~ x1 + x2", data=df_binary)
        np.testing.assert_allclose(oe_r.coefficients.values, sm_r.params, rtol=1e-8)
        np.testing.assert_allclose(oe_r.std_errors.values, sm_r.bse, rtol=1e-6)

    def test_summary_returns_string(self, df_binary):
        r = oe.probit("y ~ x1 + x2", data=df_binary)
        s = r.summary()
        assert isinstance(s, str)
        assert "PROBIT" in s

    def test_predict_class(self, df_binary):
        r = oe.probit("y ~ x1 + x2", data=df_binary)
        pred = r.predict(proba=False)
        assert set(np.unique(pred.values)).issubset({0, 1})


class TestLogitProbitContext:
    def test_context_logit(self, df_binary):
        ctx = oe.Context(df_binary)
        r = ctx.logit("y ~ x1 + x2")
        assert r.nobs == len(df_binary)

    def test_context_probit(self, df_binary):
        ctx = oe.Context(df_binary)
        r = ctx.probit("y ~ x1 + x2")
        assert r.nobs == len(df_binary)