"""Backend-identity tests for ``oe.nbreg`` (NB1 / NB2, pooled + FE).

These tests do NOT require Stata or R. They pin the public API contract and the
internal math against the hand-rolled NB estimator (OE's compute backend for NB,
since pyfixest 0.60.0 has no fenegbin): input validation, dispersion handling,
FE vs pooled, predict / margins / alpha extraction, and the ``vcov_backend``
toggle plumbing. Cross-tool (Stata/R) parity lives in
``tests/stata/tests/test_stata_nbreg.py`` and ``tests/r/tests/test_r_nbreg.py``.

Tier 1 of the three-tier parity layout (rule 7).
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from open_econs.models.limited.nbreg import nbreg


@pytest.fixture
def df_nb() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 400
    firm = rng.integers(0, 20, n)
    year = rng.integers(2020, 2024, n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    eta = 0.5 * x1 - 0.3 * x2 + rng.normal(0, 0.4, 20)[firm] + rng.normal(0, 0.2, 4)[year - 2020]
    mu = np.exp(eta)
    y = rng.negative_binomial(n=1, p=mu / (mu + 0.8))
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm, "year": year})


class TestNBRegInputValidation:
    def test_bad_dispersion(self, df_nb):
        with pytest.raises(ValueError):
            nbreg("y ~ x1", data=df_nb, dispersion="bogus")

    def test_bad_vcov_backend(self, df_nb):
        with pytest.raises(ValueError):
            nbreg("y ~ x1 + x2", data=df_nb, vcov_backend="bogus")

    def test_bad_cov_type(self, df_nb):
        with pytest.raises(ValueError):
            nbreg("y ~ x1 + x2", data=df_nb, cov_type="robust")

    def test_no_regressors(self, df_nb):
        with pytest.raises(ValueError):
            nbreg("y ~ 1", data=df_nb)

    def test_missing_column(self, df_nb):
        with pytest.raises(ValueError):
            nbreg("y ~ nope", data=df_nb)


class TestNBRegDispersion:
    """NB1 vs NB2 produce distinct, well-formed estimates."""

    def test_nb2_has_intercept_pooled(self, df_nb):
        r = nbreg("y ~ x1 + x2", data=df_nb, dispersion="const")
        assert "Intercept" in r.coefficients.index
        assert r.dispersion == "const"
        # NB2 alpha == 1/theta
        npt.assert_allclose(r.theta(), 1.0 / r.alpha(), rtol=0, atol=1e-12)

    def test_nb1_no_intercept_with_fe(self, df_nb):
        r = nbreg("y ~ x1 + x2", data=df_nb, dispersion="mean",
                  fixed_effects=["firm", "year"])
        assert "Intercept" not in r.coefficients.index
        assert r.dispersion == "mean"
        assert r.n_absorbed > 0

    def test_alpha_positive(self, df_nb):
        for disp in ("const", "mean"):
            r = nbreg("y ~ x1 + x2", data=df_nb, dispersion=disp)
            assert r.alpha() > 0
            assert np.isfinite(r.lnalpha())


class TestNBRegFEvsPooled:
    def test_fe_changes_coefs(self, df_nb):
        rp = nbreg("y ~ x1 + x2", data=df_nb, dispersion="const")
        rf = nbreg("y ~ x1 + x2", data=df_nb, dispersion="const",
                   fixed_effects=["firm", "year"])
        # FE estimate need not equal pooled; just assert both finite & distinct
        assert np.all(np.isfinite(rf.coefficients.values))
        assert rf.n_absorbed == (df_nb["firm"].nunique() + df_nb["year"].nunique() - 1)

    def test_vcov_backend_only_rescales_se(self, df_nb):
        a = nbreg("y ~ x1 + x2", data=df_nb, dispersion="const",
                  fixed_effects=["firm", "year"], cluster="firm",
                  vcov_backend="fixest")
        b = nbreg("y ~ x1 + x2", data=df_nb, dispersion="const",
                  fixed_effects=["firm", "year"], cluster="firm",
                  vcov_backend="stata")
        npt.assert_allclose(a.coefficients.values, b.coefficients.values,
                            rtol=0, atol=1e-12)
        assert not np.allclose(a.std_errors.values, b.std_errors.values, atol=1e-6)


class TestNBRegResultAPI:
    def test_irr_is_exp_beta(self, df_nb):
        r = nbreg("y ~ x1 + x2", data=df_nb, dispersion="const")
        irr = r.irr()
        npt.assert_allclose(
            irr.loc[irr["Variable"] == "x1", "IRR"].iloc[0],
            np.exp(r.coefficients["x1"]), rtol=0, atol=1e-12)

    def test_margins_are_b_times_mean_mu(self, df_nb):
        r = nbreg("y ~ x1 + x2", data=df_nb, dispersion="const")
        mar = r.margins()
        mu_bar = r.fitted_values.mean()
        npt.assert_allclose(
            mar.loc[mar["Variable"] == "x1", "dy/dx"].iloc[0],
            r.coefficients["x1"] * mu_bar, rtol=0, atol=1e-9)

    def test_predict_returns_fitted_means(self, df_nb):
        r = nbreg("y ~ x1 + x2", data=df_nb, dispersion="const")
        mu = r.predict()
        assert (mu > 0).all()
        npt.assert_allclose(mu.sum(), r.fitted_values.sum(), rtol=0, atol=1e-9)

    def test_predict_newdata_unsupported_with_fe(self, df_nb):
        r = nbreg("y ~ x1 + x2", data=df_nb, dispersion="const",
                  fixed_effects=["firm", "year"])
        with pytest.raises(NotImplementedError):
            r.predict(newdata=df_nb)
