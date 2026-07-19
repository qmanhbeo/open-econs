"""Unit / consistency tests for ``oe.robust_reg`` (no Stata/R ground truth).

These tests exercise the estimator directly: toggle validation, the
pure-Python Stata ``rreg`` default branch, robustness-weight properties
(outliers down-weighted toward zero), predict, and agreement with R
``MASS::rlm`` when R is installed.  They do NOT depend on the committed
Stata/R fixtures, so they run in any environment.

Coefficient / SE parity against R ``MASS::rlm`` (the validated ``parity="rlm"``
branch) is asserted at ``atol=1e-6`` when R is available; otherwise the test
is skipped so the suite stays green on R-less CI.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe
from open_econs.core._rlm_r import r_available


def _make_data(seed: int = 1, n_out: int = 8, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1.5, n)
    x2 = rng.normal(2, 2.0, n)
    y = 1.0 + 2.5 * x1 - 1.3 * x2 + rng.normal(0, 1.0, n)
    out = rng.choice(n, n_out, replace=False)
    y[out] += rng.choice([-1, 1], n_out) * rng.uniform(8, 15, n_out)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


class TestRobustRegToggleValidation:
    """Invalid toggles must raise ValueError (rule: validate toggles)."""

    def setup_method(self):
        self.df = _make_data()

    def test_bad_method(self):
        with pytest.raises(ValueError):
            oe.robust_reg("y ~ x1 + x2", data=self.df, method="lts")

    def test_bad_parity(self):
        with pytest.raises(ValueError):
            oe.robust_reg("y ~ x1 + x2", data=self.df, parity="splus")

    def test_bad_vcov(self):
        with pytest.raises(ValueError):
            oe.robust_reg("y ~ x1 + x2", data=self.df, vcov="hc1")

    def test_valid_toggles_ok(self):
        for parity in ("stata", "rlm"):
            for method in ("mm", "huber"):
                for vcov in (None, "stata", "rlm"):
                    if parity == "rlm" and not r_available():
                        continue
                    r = oe.robust_reg(
                        "y ~ x1 + x2", data=self.df, parity=parity,
                        method=method, vcov=vcov,
                    )
                    assert r.parity == parity
                    assert r.method == method
                    assert r.vcov == (vcov if vcov is not None else parity)


class TestRobustRegWeightProperties:
    """Bisquare robustness weights live in [0, 1]; outliers -> ~0."""

    def setup_method(self):
        self.df = _make_data()

    def test_weights_in_unit_interval(self):
        r = oe.robust_reg("y ~ x1 + x2", data=self.df, parity="stata")
        w = r.weights.values
        assert np.all(w >= -1e-12)
        assert np.all(w <= 1.0 + 1e-12)

    def test_outliers_downweighted(self):
        # Inject an extreme outlier.  Stata rreg.ado drops obs with Cook's D > 1
        # (so the weight at that obs is excluded -> NaN), otherwise the bisquare
        # collapses its weight toward 0.  Either way the outlier must not keep a
        # material weight, and the fit must not be driven by it.
        df = self.df.copy()
        df.loc[0, "y"] = df.loc[0, "y"] + 1e4
        r = oe.robust_reg("y ~ x1 + x2", data=df, parity="stata")
        w0 = r.weights.iloc[0]
        assert (np.isnan(w0) or w0 < 1e-3)
        # The outlier must not pull the coefficient off the clean-data fit.
        r_clean = oe.robust_reg("y ~ x1 + x2", data=self.df, parity="stata")
        assert np.allclose(r.coefficients.values, r_clean.coefficients.values, atol=1e-2)

    def test_inliers_keep_high_weight(self):
        r = oe.robust_reg("y ~ x1 + x2", data=self.df, parity="stata")
        # Bulk of weights should be large (well-fit points near 1).
        assert np.median(r.weights.values) > 0.8


class TestRobustRegCoefVsRLM:
    """Coefficients must match R MASS::rlm to 1e-6 when R is available."""

    @pytest.mark.skipif(not r_available(), reason="R/MASS::rlm not installed")
    def test_mm_coef_matches_rlm(self):
        df = _make_data()
        r = oe.robust_reg("y ~ x1 + x2", data=df, method="mm", parity="rlm")
        from open_econs.core._rlm_r import rlm_fit
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "inp.csv"
            df.to_csv(p, index=False)
            fit = rlm_fit("y ~ x1 + x2", str(p), method="MM", acc=1e-6)
        npt.assert_allclose(r.coefficients.values, np.asarray(fit["b"]), atol=1e-6)

    @pytest.mark.skipif(not r_available(), reason="R/MASS::rlm not installed")
    def test_rlm_vcov_matches_rlm(self):
        df = _make_data()
        r = oe.robust_reg("y ~ x1 + x2", data=df, method="mm", parity="rlm")
        from open_econs.core._rlm_r import rlm_fit
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "inp.csv"
            df.to_csv(p, index=False)
            fit = rlm_fit("y ~ x1 + x2", str(p), method="MM")
        se_r = np.sqrt(np.diag(np.asarray(fit["V"])))
        npt.assert_allclose(r.std_errors.values, se_r, atol=1e-6)

    @pytest.mark.skipif(not r_available(), reason="R/MASS::rlm not installed")
    def test_huber_method_runs(self):
        df = _make_data()
        r = oe.robust_reg("y ~ x1 + x2", data=df, method="huber", parity="rlm")
        assert r.method == "huber"
        assert np.isfinite(r.coefficients.values).all()


class TestRobustRegStataDefault:
    """The Stata parity branch runs without R and produces sane output."""

    def setup_method(self):
        self.df = _make_data()

    def test_default_is_stata(self):
        r = oe.robust_reg("y ~ x1 + x2", data=self.df)
        assert r.parity == "stata"
        assert r.vcov == "stata"

    def test_coef_finite_and_weighted(self):
        r = oe.robust_reg("y ~ x1 + x2", data=self.df, parity="stata")
        assert np.all(np.isfinite(r.coefficients.values))
        # robust coef should be less sensitive to outliers than OLS
        df_clean = self.df.copy()
        df_clean.loc[0, "y"] += 1e4
        r_out = oe.robust_reg("y ~ x1 + x2", data=df_clean, parity="stata")
        # The intercept should not shift by more than a few units.
        assert abs(r_out.coefficients["(Intercept)"] - r.coefficients["(Intercept)"]) < 2.0


class TestRobustRegInterface:
    """Result-object interface: predict, tidy, summary, weights/scale attrs."""

    def setup_method(self):
        self.df = _make_data()

    def test_predict_in_sample(self):
        r = oe.robust_reg("y ~ x1 + x2", data=self.df, parity="stata")
        fitted = r.predict()
        npt.assert_allclose(fitted.values, r.fitted_values.values, atol=1e-8)

    def test_predict_newdata(self):
        r = oe.robust_reg("y ~ x1 + x2", data=self.df, parity="stata")
        new = pd.DataFrame({"x1": [0.0, 1.0], "x2": [2.0, -1.0]})
        pred = r.predict(newdata=new)
        assert len(pred) == 2
        expected = (
            r.coefficients["(Intercept)"]
            + new["x1"].values * r.coefficients["x1"]
            + new["x2"].values * r.coefficients["x2"]
        )
        npt.assert_allclose(pred.values, expected, atol=1e-8)

    def test_tidy_columns(self):
        r = oe.robust_reg("y ~ x1 + x2", data=self.df, parity="stata")
        t = r.tidy()
        assert list(t.columns) == [
            "Variable", "Coef", "Std Err", "t", "P>|t|", "0.025", "0.975",
        ]

    def test_summary_runs(self):
        r = oe.robust_reg("y ~ x1 + x2", data=self.df, parity="stata")
        s = r.summary()
        assert "Robust Regression" in s
        assert "M-estimate scale" in s

    def test_immutability(self):
        r = oe.robust_reg("y ~ x1 + x2", data=self.df, parity="stata")
        with pytest.raises(AttributeError):
            r.coefficients = r.coefficients

    def test_weights_and_scale_present(self):
        r = oe.robust_reg("y ~ x1 + x2", data=self.df, parity="stata")
        assert r.scale > 0
        assert len(r.weights) == len(self.df)
