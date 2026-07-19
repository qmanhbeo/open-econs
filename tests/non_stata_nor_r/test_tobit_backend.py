"""Backend-identity tests for ``oe.tobit`` (hand-rolled censored-normal MLE).

These tests do NOT require Stata or R. They pin the public API contract and the
internal math of OE's own Tobit implementation: input validation, censoring
limits, the latent/observed/probability prediction algebra, ``sigma`` extraction,
the ``Log(scale)`` vs ``sigma`` crosswalk, and margins. Cross-tool (Stata/R)
parity lives in ``tests/stata/tests/test_stata_tobit.py`` and
``tests/r/tests/test_r_tobit.py``.

Tier 1 of the three-tier parity layout (rule 7).
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe
from open_econs.models.limited.tobit import tobit


@pytest.fixture
def df_tobit() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 500
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    ystar = 0.5 + 0.8 * x1 - 0.5 * x2 + 0.3 * x3 + rng.normal(0, 1, n)
    y = np.maximum(0.0, ystar)            # left-censored at 0
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


class TestTobitInputValidation:
    """Constructor / kwarg guards."""

    def test_bad_cov_type(self, df_tobit):
        with pytest.raises(ValueError):
            tobit("y ~ x1 + x2", data=df_tobit, cov_type="HC9")

    def test_missing_column(self, df_tobit):
        with pytest.raises(ValueError):
            tobit("y ~ nope", data=df_tobit)

    def test_no_left_censoring_none_ok(self, df_tobit):
        # left=None disables left censoring (OLS-equivalent MLE).
        r = tobit("y ~ x1 + x2", data=df_tobit, left=None)
        assert np.isfinite(r.sigma)
        assert r.n_left == 0

    def test_bad_cluster_column(self, df_tobit):
        with pytest.raises(ValueError):
            tobit("y ~ x1 + x2", data=df_tobit, cluster="nope")


class TestTobitCensoringLimits:
    """Censoring limit plumbing (left / right / none)."""

    def test_left_censoring_count(self, df_tobit):
        r = tobit("y ~ x1 + x2 + x3", data=df_tobit, left=0)
        assert r.n_left == int((df_tobit["y"] <= 0).sum())
        assert r.left == 0.0

    def test_right_censoring(self, df_tobit):
        rng = np.random.default_rng(3)
        ystar = 0.5 + 0.8 * df_tobit["x1"] - 0.5 * df_tobit["x2"]
        yr = np.minimum(2.0, ystar + rng.normal(0, 1, len(df_tobit)))
        df = df_tobit.assign(yr=yr)
        r = tobit("yr ~ x1 + x2", data=df, left=None, right=2.0)
        assert r.n_right == int((yr >= 2.0).sum())
        assert r.right == 2.0

    def test_left_none_right_none_is_ols(self, df_tobit):
        # With no censoring, Tobit MLE = OLS on the latent (here observed) outcome.
        import open_econs as oe_mod
        r_tobit = tobit("y ~ x1 + x2 + x3", data=df_tobit, left=None)
        from open_econs.models.linear.ols import ols
        r_ols = ols("y ~ x1 + x2 + x3", data=df_tobit)
        npt.assert_allclose(
            r_tobit.coefficients.values, r_ols.coefficients.values, rtol=0, atol=1e-4
        )


class TestTobitPredictionAlgebra:
    """E[y*] vs E[y|y>0] vs P(y>0) algebra and crosswalk."""

    def test_log_scale_is_log_sigma(self, df_tobit):
        r = tobit("y ~ x1 + x2 + x3", data=df_tobit, left=0)
        npt.assert_allclose(r.log_scale, np.log(r.sigma), rtol=0, atol=1e-12)

    def test_predict_ystar_is_linear_predictor(self, df_tobit):
        r = tobit("y ~ x1 + x2 + x3", data=df_tobit, left=0)
        ys = r.predict(type="ystar")
        expected = (
            r.coefficients["Intercept"]
            + df_tobit["x1"].values * r.coefficients["x1"]
            + df_tobit["x2"].values * r.coefficients["x2"]
            + df_tobit["x3"].values * r.coefficients["x3"]
        )
        npt.assert_allclose(ys.values, expected, rtol=0, atol=1e-9)

    def test_predict_y_between_limits(self, df_tobit):
        r = tobit("y ~ x1 + x2 + x3", data=df_tobit, left=0)
        ey = r.predict(type="y")
        # E[y|x] must be >= left limit (0) and <= right limit (inf).
        assert (ey.values >= -1e-9).all()

    def test_predict_pr_gt0_in_unit_interval(self, df_tobit):
        r = tobit("y ~ x1 + x2 + x3", data=df_tobit, left=0)
        pr = r.predict(type="pr_gt0")
        assert (pr.values >= 0.0).all() and (pr.values <= 1.0).all()

    def test_predict_out_of_sample(self, df_tobit):
        r = tobit("y ~ x1 + x2 + x3", data=df_tobit, left=0)
        new = df_tobit.iloc[:5].copy()
        ey_new = r.predict(newdata=new, type="y")
        assert len(ey_new) == 5
        ys_new = r.predict(newdata=new, type="ystar")
        npt.assert_allclose(
            ys_new.values,
            r.coefficients["Intercept"]
            + new["x1"].values * r.coefficients["x1"]
            + new["x2"].values * r.coefficients["x2"]
            + new["x3"].values * r.coefficients["x3"],
            rtol=0, atol=1e-9,
        )

    def test_sigma_extraction_and_vcov_full(self, df_tobit):
        r = tobit("y ~ x1 + x2 + x3", data=df_tobit, left=0)
        assert "sigma" in r.vcov(full=True).index
        v = r.vcov(full=True).loc["sigma", "sigma"]
        assert v >= 0.0


class TestTobitMargins:
    """Margins on E[y|x] = beta * P(y > left)."""

    def test_margins_are_beta_times_pr_mean(self, df_tobit):
        r = tobit("y ~ x1 + x2 + x3", data=df_tobit, left=0)
        m = r.margins()
        pr_mean = r.fitted_pr.mean()
        for name in r.coefficients.index:
            ame = m.loc[m["Variable"] == name, "dy/dx"].iloc[0]
            npt.assert_allclose(
                ame, r.coefficients[name] * pr_mean, rtol=0, atol=1e-9
            )
