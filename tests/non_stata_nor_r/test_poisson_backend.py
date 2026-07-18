"""Backend-identity tests for ``oe.poisson`` (FE-backed Poisson PPML).

These tests do NOT require Stata or R. They pin the public API contract and the
internal math against ``pyfixest.fepois`` (OE's compute backend for count FE):
input validation, FE absorption, IRR / margins / predict algebra, and the
``vcov_backend`` toggle plumbing. Cross-tool (Stata/R) parity livest in
``tests/stata/tests/test_stata_poisson.py`` and ``tests/r/tests/test_r_poisson.py``.

Tier 1 of the three-tier parity layout (rule 7).
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe
from open_econs.models.limited.poisson import poisson


@pytest.fixture
def df_poisson() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 300
    firm = rng.integers(0, 15, n)
    year = rng.integers(2020, 2024, n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    eta = 0.4 * x1 - 0.2 * x2 + rng.normal(0, 0.4, 15)[firm] + rng.normal(0, 0.2, 4)[year - 2020]
    y = rng.poisson(np.exp(eta))
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm, "year": year})


class TestPoissonInputValidation:
    """Constructor / kwarg guards."""

    def test_requires_fe(self, df_poisson):
        with pytest.raises(ValueError):
            poisson("y ~ x1", data=df_poisson)

    def test_entity_and_fixed_effects_mutually_exclusive(self, df_poisson):
        with pytest.raises(ValueError):
            poisson("y ~ x1 + x2", data=df_poisson, fixed_effects=["firm"],
                    entity="firm")

    def test_bad_vcov_backend(self, df_poisson):
        with pytest.raises(ValueError):
            poisson("y ~ x1 + x2", data=df_poisson, fixed_effects=["firm"],
                    vcov_backend="bogus")

    def test_bad_cov_type(self, df_poisson):
        with pytest.raises(ValueError):
            poisson("y ~ x1 + x2", data=df_poisson, fixed_effects=["firm"],
                    cov_type="HC2")

    def test_missing_column(self, df_poisson):
        with pytest.raises(ValueError):
            poisson("y ~ nope", data=df_poisson, fixed_effects=["firm"])


class TestPoissonBackendIdentity:
    """OE result equals pyfixest.fepois directly (same compute backend)."""

    def test_coef_equals_pyfixest(self, df_poisson):
        import pyfixest as pf
        r = poisson("y ~ x1 + x2", data=df_poisson, fixed_effects=["firm", "year"])
        m = pf.fepois("y ~ x1 + x2 | firm + year", data=df_poisson, vcov="iid")
        npt.assert_allclose(r.coefficients["x1"], m.coef()["x1"], rtol=0, atol=1e-9)
        npt.assert_allclose(r.coefficients["x2"], m.coef()["x2"], rtol=0, atol=1e-9)

    def test_cluster_se_equals_pyfixest(self, df_poisson):
        import pyfixest as pf
        r = poisson("y ~ x1 + x2", data=df_poisson, fixed_effects=["firm", "year"],
                    cluster="firm")
        m = pf.fepois("y ~ x1 + x2 | firm + year", data=df_poisson, vcov={"CRV1": "firm"})
        npt.assert_allclose(r.std_errors["x1"], m.se()["x1"], rtol=0, atol=1e-9)

    def test_multiway_cluster(self, df_poisson):
        # N-way FE + multi-way cluster must not raise and must return FE.
        r = poisson("y ~ x1 + x2", data=df_poisson, fixed_effects=["firm", "year"],
                    cluster=["firm", "year"])
        assert r.n_absorbed > 0
        assert r.fixed_effects == ["firm", "year"]

    def test_vcov_backend_only_rescales_se(self, df_poisson):
        a = poisson("y ~ x1 + x2", data=df_poisson, fixed_effects=["firm", "year"],
                    cluster="firm", vcov_backend="fixest")
        b = poisson("y ~ x1 + x2", data=df_poisson, fixed_effects=["firm", "year"],
                    cluster="firm", vcov_backend="stata")
        # point estimates identical across toggles
        npt.assert_allclose(a.coefficients.values, b.coefficients.values, rtol=0, atol=1e-12)
        # SEs differ (fixest vs stata convention) -> toggle is doing its job
        assert not np.allclose(a.std_errors.values, b.std_errors.values, atol=1e-6)


class TestPoissonResultAPI:
    """IRR / margins / predict algebra."""

    def test_irr_is_exp_beta(self, df_poisson):
        r = poisson("y ~ x1 + x2", data=df_poisson, fixed_effects=["firm", "year"])
        irr = r.irr()
        npt.assert_allclose(irr.loc[irr["Variable"] == "x1", "IRR"].iloc[0],
                            np.exp(r.coefficients["x1"]), rtol=0, atol=1e-12)
        # delta-method SE: irr_se = irr * se
        se = irr.loc[irr["Variable"] == "x1", "Std Err"].iloc[0]
        npt.assert_allclose(se,
                            np.exp(r.coefficients["x1"]) * r.std_errors["x1"],
                            rtol=0, atol=1e-12)

    def test_margins_are_b_times_mean_mu(self, df_poisson):
        r = poisson("y ~ x1 + x2", data=df_poisson, fixed_effects=["firm", "year"])
        mar = r.margins()
        mu_bar = r.fitted_values.mean()
        npt.assert_allclose(mar.loc[mar["Variable"] == "x1", "dy/dx"].iloc[0],
                            r.coefficients["x1"] * mu_bar, rtol=0, atol=1e-9)

    def test_predict_returns_fitted_means(self, df_poisson):
        r = poisson("y ~ x1 + x2", data=df_poisson, fixed_effects=["firm", "year"])
        mu = r.predict()
        # fitted means should be positive and sum close to observed
        assert (mu > 0).all()
        npt.assert_allclose(mu.sum(), r.fitted_values.sum(), rtol=0, atol=1e-9)

    def test_irr_conf_int_is_exponentiated(self, df_poisson):
        r = poisson("y ~ x1 + x2", data=df_poisson, fixed_effects=["firm", "year"])
        irr = r.irr()
        lo = irr.loc[irr["Variable"] == "x1", "0.025"].iloc[0]
        hi = irr.loc[irr["Variable"] == "x1", "0.975"].iloc[0]
        npt.assert_allclose(lo, np.exp(r.conf_int.loc["x1", "lower"]), rtol=0, atol=1e-10)
        npt.assert_allclose(hi, np.exp(r.conf_int.loc["x1", "upper"]), rtol=0, atol=1e-10)
