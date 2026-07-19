"""Unit / consistency tests for quantile_reg() (no Stata or R required).

Covers the L1 optimality property of median regression, coefficient agreement
with statsmodels' quantile_regression at the median (where the solution is
unique), toggle validation, bootstrap reproducibility, and predict().
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe


@pytest.fixture(scope="module")
def df_qr() -> pd.DataFrame:
    rng = np.random.default_rng(20260719)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.uniform(-2, 2, n)
    eps = rng.standard_t(5, n) * (1 + 0.3 * np.abs(x1))
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + eps
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


class TestBasics:
    def test_import(self):
        assert hasattr(oe, "quantile_reg")
        assert hasattr(oe, "QuantileResult")

    def test_default_is_median(self, df_qr):
        r = oe.quantile_reg("y ~ x1 + x2", df_qr)
        assert r.quantile() == 0.5
        assert r.method == "qreg"

    def test_coef_names_and_shapes(self, df_qr):
        r = oe.quantile_reg("y ~ x1 + x2", df_qr)
        assert list(r.coefficients.index) == ["Intercept", "x1", "x2"]
        assert r.std_errors.shape == (3,)
        assert r.tidy().shape == (3, 7)

    def test_result_is_immutable(self, df_qr):
        r = oe.quantile_reg("y ~ x1 + x2", df_qr)
        with pytest.raises(AttributeError):
            r.tau = 0.9


class TestMedianL1Property:
    """Median regression minimises the sum of absolute residuals; the residual
    signs must (nearly) balance at the optimum, and the median fit must beat OLS
    on the L1 loss."""

    def test_l1_beats_ols(self, df_qr):
        qr = oe.quantile_reg("y ~ x1 + x2", df_qr, tau=0.5)
        ols = oe.ols("y ~ x1 + x2", df_qr, cov_type="nonrobust")
        l1_qr = np.sum(np.abs(qr.residuals.values))
        l1_ols = np.sum(np.abs(ols.residuals.values))
        assert l1_qr <= l1_ols + 1e-8

    def test_residual_sign_balance(self, df_qr):
        qr = oe.quantile_reg("y ~ x1 + x2", df_qr, tau=0.5)
        resid = qr.residuals.values
        n = len(resid)
        k = 3
        n_neg = int(np.sum(resid < 0))
        # At a QR vertex, at most k residuals are exactly zero and the negative
        # share is within k of tau*n.
        assert abs(n_neg - 0.5 * n) <= k

    def test_tau_quantile_share(self, df_qr):
        # For tau=0.25 the share of negative residuals is ~tau (within k obs).
        qr = oe.quantile_reg("y ~ x1 + x2", df_qr, tau=0.25)
        resid = qr.residuals.values
        n = len(resid)
        n_neg = int(np.sum(resid < 0))
        assert abs(n_neg - 0.25 * n) <= 3


class TestCoefVsStatsmodels:
    """At the median the QR solution is unique, so statsmodels' interior-point
    QuantReg lands on the same vertex as our BR-simplex LP."""

    def test_median_matches_statsmodels(self, df_qr):
        # NOTE: statsmodels' QuantReg uses an interior-point/IRLS solver that
        # differs from the Barrodale-Roberts simplex by ~1e-5 even at the
        # unique median optimum (a documented solver, NOT a convention,
        # divergence).  Stata qreg / R rq(method="br") agree with our LP solve
        # to machine precision (see parity tests).  Tolerance is therefore
        # widened for statsmodels only.
        import statsmodels.formula.api as smf

        r = oe.quantile_reg("y ~ x1 + x2", df_qr, tau=0.5)
        m = smf.quantreg("y ~ x1 + x2", df_qr).fit(q=0.5)
        npt.assert_allclose(
            r.coefficients.values,
            [m.params["Intercept"], m.params["x1"], m.params["x2"]],
            atol=1e-5,
        )


class TestToggleValidation:
    @pytest.mark.parametrize("bad_tau", [0.0, 1.0, -0.1, 1.5, 2.0])
    def test_tau_out_of_range_rejected(self, df_qr, bad_tau):
        with pytest.raises(ValueError, match="tau"):
            oe.quantile_reg("y ~ x1 + x2", df_qr, tau=bad_tau)

    def test_bad_method_rejected(self, df_qr):
        with pytest.raises(ValueError, match="method"):
            oe.quantile_reg("y ~ x1 + x2", df_qr, method="lasso")

    def test_bad_se_method_rejected(self, df_qr):
        with pytest.raises(ValueError, match="se_method"):
            oe.quantile_reg("y ~ x1 + x2", df_qr, se_method="rank")

    def test_bad_cov_type_rejected(self, df_qr):
        with pytest.raises(ValueError, match="cov_type"):
            oe.quantile_reg("y ~ x1 + x2", df_qr, cov_type="HC1")

    def test_missing_column_rejected(self, df_qr):
        with pytest.raises(Exception):
            oe.quantile_reg("y ~ x1 + nope", df_qr)


class TestSeMethodBranches:
    def test_two_se_methods_differ(self, df_qr):
        r_stata = oe.quantile_reg("y ~ x1 + x2", df_qr, se_method="stata")
        r_ker = oe.quantile_reg("y ~ x1 + x2", df_qr, se_method="ker")
        # Same coefficients, different SEs.
        npt.assert_allclose(
            r_stata.coefficients.values, r_ker.coefficients.values, atol=1e-10
        )
        gap = np.max(np.abs(r_stata.std_errors.values - r_ker.std_errors.values))
        assert gap > 1e-4

    def test_all_se_positive(self, df_qr):
        for sem in ("stata", "ker"):
            r = oe.quantile_reg("y ~ x1 + x2", df_qr, se_method=sem)
            assert np.all(r.std_errors.values > 0)


class TestBootstrap:
    def test_bootstrap_reproducible_with_seed(self, df_qr):
        r1 = oe.quantile_reg("y ~ x1 + x2", df_qr, method="bsqreg", seed=42, reps=25)
        r2 = oe.quantile_reg("y ~ x1 + x2", df_qr, method="bsqreg", seed=42, reps=25)
        npt.assert_allclose(r1.std_errors.values, r2.std_errors.values, atol=1e-12)

    def test_bootstrap_coef_matches_analytic(self, df_qr):
        # The point estimate does not depend on the VCE method.
        r_bs = oe.quantile_reg("y ~ x1 + x2", df_qr, method="bsqreg", seed=1)
        r_an = oe.quantile_reg("y ~ x1 + x2", df_qr, method="qreg")
        npt.assert_allclose(
            r_bs.coefficients.values, r_an.coefficients.values, atol=1e-10
        )

    def test_sqreg_single_tau_equals_bsqreg(self, df_qr):
        r_sq = oe.quantile_reg("y ~ x1 + x2", df_qr, method="sqreg", seed=7, reps=30)
        r_bs = oe.quantile_reg("y ~ x1 + x2", df_qr, method="bsqreg", seed=7, reps=30)
        npt.assert_allclose(r_sq.std_errors.values, r_bs.std_errors.values, atol=1e-12)


class TestPredict:
    def test_predict_insample(self, df_qr):
        r = oe.quantile_reg("y ~ x1 + x2", df_qr)
        pred = r.predict()
        npt.assert_allclose(pred.values, r.fitted_values.values, atol=1e-12)

    def test_predict_newdata(self, df_qr):
        r = oe.quantile_reg("y ~ x1 + x2", df_qr)
        newd = pd.DataFrame({"x1": [0.0, 1.0], "x2": [0.0, -1.0]})
        pred = r.predict(newd)
        b = r.coefficients
        expected = [
            b["Intercept"],
            b["Intercept"] + b["x1"] - b["x2"],
        ]
        npt.assert_allclose(pred.values, expected, atol=1e-10)


class TestMonotonicity:
    def test_higher_tau_higher_intercept(self, df_qr):
        # For this DGP the conditional-quantile intercept increases with tau.
        icepts = [
            oe.quantile_reg("y ~ x1 + x2", df_qr, tau=t).coefficients["Intercept"]
            for t in (0.1, 0.5, 0.9)
        ]
        assert icepts[0] < icepts[1] < icepts[2]
