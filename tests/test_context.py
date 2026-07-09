import numpy as np
import pandas as pd

import open_econs as oe


class TestContext:
    def test_context_ols(self, df_ols):
        ctx = oe.Context(df_ols)
        r = ctx.ols("income ~ education + age")
        assert r.nobs == len(df_ols)

    def test_context_oaxaca(self, df_oaxaca):
        ctx = oe.Context(df_oaxaca)
        d = ctx.oaxaca("income ~ education + age + female", by="female")
        assert d.nobs == len(df_oaxaca)

    def test_repr(self, df_ols):
        ctx = oe.Context(df_ols)
        s = repr(ctx)
        assert "Context" in s
        assert str(df_ols.shape[0]) in s

    def test_context_logit(self, df_ols):
        df = df_ols.copy()
        df["y_bin"] = (df["income"] > df["income"].median()).astype(float)
        ctx = oe.Context(df)
        r = ctx.logit("y_bin ~ education + age")
        assert r.nobs == len(df)

    def test_context_probit(self, df_ols):
        df = df_ols.copy()
        df["y_bin"] = (df["income"] > df["income"].median()).astype(float)
        ctx = oe.Context(df)
        r = ctx.probit("y_bin ~ education + age")
        assert r.nobs == len(df)

    def test_vif_returns_series(self, df_ols):
        ctx = oe.Context(df_ols)
        v = ctx.vif("income ~ education + age")
        assert isinstance(v, pd.Series)
        assert v.name == "VIF"

    def test_vif_values_above_one(self, df_ols):
        ctx = oe.Context(df_ols)
        v = ctx.vif("income ~ education + age")
        for val in v.values:
            assert val > 0.5

    def test_vif_rhs_only(self, df_ols):
        ctx = oe.Context(df_ols)
        v = ctx.vif("education + age")
        assert isinstance(v, pd.Series)
        assert len(v) >= 2

    def test_vif_on_collinear_design(self, df_ols):
        df = df_ols.copy()
        df["edu2"] = df["education"] * 2 + 0.01 * np.random.normal(0, 1, len(df))
        ctx = oe.Context(df)
        v = ctx.vif("y ~ education + edu2")
        high_vif = v.max()
        assert high_vif > 5 or (v > 5).any()

    def test_vif_index_names(self, df_ols):
        ctx = oe.Context(df_ols)
        v = ctx.vif("education + age")
        assert v.index.name == "Variable"