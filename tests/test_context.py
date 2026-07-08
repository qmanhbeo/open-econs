import pandas as pd
import pytest

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