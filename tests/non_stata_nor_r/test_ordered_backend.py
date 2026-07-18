"""Backend-identity tests for ``oe.ologit`` / ``oe.oprobit`` (tier 1).

These tests do NOT require Stata or R. They pin the public API contract and the
internal math against ``statsmodels.miscmodels.ordinal_model.OrderedModel`` (OE's
compute backend for ordered models): input validation, cutpoint sign/algebra,
predict class-probability algebra, and margins. Cross-tool (Stata/R) parity
lives in ``tests/stata/tests/test_stata_ordered.py`` and
``tests/r/tests/test_r_ordered.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from open_econs.models.limited.ordered import ologit, oprobit, _sandwich_cov


@pytest.fixture
def df_ordered() -> pd.DataFrame:
    rng = np.random.default_rng(1234)
    n = 600
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    eta = 0.7 * x1 - 0.5 * x2 + 0.3 * x3
    y_star = eta + rng.normal(0, 1, n)
    y = np.digitize(y_star, [-0.8, 0.2, 1.0]).astype(int)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


class TestOrderedInputValidation:
    def test_binary_depvar_rejected(self, df_ordered):
        d = df_ordered.assign(y2=(df_ordered["y"] > 1).astype(int))
        with pytest.raises(ValueError):
            ologit("y2 ~ x1 + x2 + x3", data=d)

    def test_missing_column(self, df_ordered):
        with pytest.raises(ValueError):
            ologit("y ~ nope", data=df_ordered)

    def test_bad_cov_type(self, df_ordered):
        with pytest.raises(ValueError):
            ologit("y ~ x1 + x2 + x3", data=df_ordered, cov_type="HC9")

    def test_formula_with_intercept_ok(self, df_ordered):
        # intercept in the formula must be silently dropped (OrderedModel supplies thresholds)
        r = ologit("y ~ 1 + x1 + x2 + x3", data=df_ordered)
        assert set(r.coefficients.index) == {"x1", "x2", "x3"}


class TestOrderedBackendIdentity:
    def test_coef_equals_statsmodels(self, df_ordered):
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        yc = pd.Series(pd.Categorical(df_ordered["y"].astype(str),
                                      categories=["0", "1", "2", "3"], ordered=True))
        m = OrderedModel(yc, df_ordered[["x1", "x2", "x3"]], distr="logit")
        sm_fit = m.fit(disp=False, method="nm")
        r = ologit("y ~ x1 + x2 + x3", data=df_ordered)
        # OE polishes past statsmodels' default stop; coefs equal to 1e-4 (optimizer precision)
        npt.assert_allclose(r.coefficients["x1"], sm_fit.params["x1"], rtol=0, atol=1e-4)
        npt.assert_allclose(r.coefficients["x3"], sm_fit.params["x3"], rtol=0, atol=1e-4)

    def test_cutpoints_increasing_and_match_transform(self, df_ordered):
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        yc = pd.Series(pd.Categorical(df_ordered["y"].astype(str),
                                      categories=["0", "1", "2", "3"], ordered=True))
        m = OrderedModel(yc, df_ordered[["x1", "x2", "x3"]], distr="logit")
        sm_fit = m.fit(disp=False, method="nm")
        p = np.asarray(sm_fit.params)
        sm_cuts = m.transform_threshold_params(p)[1:-1]
        r = ologit("y ~ x1 + x2 + x3", data=df_ordered)
        # OE cutpoints equal the statsmodels cumulative thresholds (Stata convention)
        npt.assert_allclose(r.cutpoints.values, sm_cuts, rtol=0, atol=1e-4)
        # and they must be strictly increasing
        assert (np.diff(r.cutpoints.values) > 0).all()

    def test_cutpoint_sign_matches_stata_not_negated(self, df_ordered):
        # ROOT CAUSE (methodology/limited/ordered.md): statsmodels, Stata, and
        # R polr ALL store cumulative thresholds with the SAME sign
        # (P(Y<=j) = F(cut_j - xb)). OE stores cutpoints in that convention,
        # so they match Stata's ologit cut* directly (no negation).
        r = ologit("y ~ x1 + x2 + x3", data=df_ordered)
        from tests.stata.stata_runner import read_stata
        S = read_stata("ordered")
        # same input fixture -> cutpoints have the same sign as Stata's
        assert np.sign(r.cutpoints["cut1"]) == np.sign(S["ologit_cut1"])


class TestOrderedResultAPI:
    def test_predict_probs_sum_to_one(self, df_ordered):
        r = ologit("y ~ x1 + x2 + x3", data=df_ordered)
        P = r.predict(type="probs")
        npt.assert_allclose(P.sum(axis=1).values, 1.0, rtol=0, atol=1e-12)
        assert list(P.columns) == ["0", "1", "2", "3"]

    def test_predict_class_is_argmax(self, df_ordered):
        r = ologit("y ~ x1 + x2 + x3", data=df_ordered)
        P = r.predict(type="probs")
        cls = r.predict(type="class")
        expected = P.values.argmax(axis=1)
        npt.assert_array_equal(cls.values, expected)

    def test_predict_newdata_probs(self, df_ordered):
        r = ologit("y ~ x1 + x2 + x3", data=df_ordered)
        nd = df_ordered.iloc[:5].drop(columns=["y"])
        P = r.predict(newdata=nd, type="probs")
        assert P.shape == (5, 4)
        npt.assert_allclose(P.sum(axis=1).values, 1.0, rtol=0, atol=1e-12)

    def test_margins_sum_to_zero(self, df_ordered):
        r = ologit("y ~ x1 + x2 + x3", data=df_ordered)
        mg = r.margins()
        for v in ["x1", "x2", "x3"]:
            s = mg.loc[mg["Variable"] == v, "dy/dx"].sum()
            npt.assert_allclose(s, 0.0, atol=1e-9)

    def test_tidy_has_cutpoints_attr(self, df_ordered):
        r = ologit("y ~ x1 + x2 + x3", data=df_ordered)
        assert "cut1" in r.cutpoints.index
        assert r.cutpoints.shape[0] == 3
        assert r.coefficients.shape[0] == 3

    def test_ologit_and_oprobit_differ(self, df_ordered):
        rl = ologit("y ~ x1 + x2 + x3", data=df_ordered)
        rp = oprobit("y ~ x1 + x2 + x3", data=df_ordered)
        assert rl.distr == "logit"
        assert rp.distr == "probit"
        # coefficients differ between links
        assert not np.allclose(rl.coefficients.values, rp.coefficients.values, atol=1e-3)


class TestOrderedSandwich:
    def test_sandwich_hc1_scales_hc0(self, df_ordered):
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        yc = pd.Series(pd.Categorical(df_ordered["y"].astype(str),
                                      categories=["0", "1", "2", "3"], ordered=True))
        m = OrderedModel(yc, df_ordered[["x1", "x2", "x3"]], distr="logit")
        fit = m.fit(disp=False, method="nm")
        p = np.asarray(fit.params)
        bread = np.linalg.inv(-m.hessian(p))
        cov0 = _sandwich_cov(m, p, bread, "HC0")
        cov1 = _sandwich_cov(m, p, bread, "HC1")
        n = m.nobs
        k = p.shape[0]
        # HC1 = (n/(n-k)) * HC0 on the diagonal
        ratio = np.sqrt(np.diag(cov1) / np.diag(cov0))
        npt.assert_allclose(ratio, np.sqrt(n / (n - k)), rtol=0, atol=1e-12)
