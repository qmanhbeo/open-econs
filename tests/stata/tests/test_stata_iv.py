"""Stata parity tests for IV / 2SLS."""

from __future__ import annotations

import numpy.testing as npt
import pytest

import open_econs as oe

from ..stata_runner import read_stata

pytestmark = pytest.mark.stata

S_IV = read_stata("iv_basic")


class TestIVBasic:
    @pytest.fixture(autouse=True)
    def _run(self, df_iv):
        self.s = S_IV
        self.oe_r = oe.iv("y ~ x2 | x ~ z", data=df_iv, cov_type="nonrobust")

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_int"], self.s["b_x2"], self.s["b_x"]],
                            rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_int"], self.s["se_x2"], self.s["se_x"]],
                            rtol=1e-6)

    def test_nobs(self):
        assert self.oe_r.nobs == int(self.s["N"])


S_IV_CLUSTER = read_stata("iv_cluster")
S_IV_ROBUST = read_stata("iv_robust")
S_IV_CLUSTER_PANEL = read_stata("iv_cluster_panel")
S_IV_FE = read_stata("iv_fe")


class TestIVCluster:
    """Single-way cluster-robust IV-2SLS matches Stata ``ivregress, vce(cluster)``."""

    @pytest.fixture(autouse=True)
    def _run(self, df_iv_cluster):
        self.s = S_IV_CLUSTER
        self.oe_r = oe.iv("y ~ w | x ~ z", data=df_iv_cluster, cluster="firm")

    def test_coefficients(self):
        npt.assert_allclose(
            self.oe_r.coefficients.values,
            [self.s["b_int"], self.s["b_w"], self.s["b_x"]],
            rtol=1e-6,
        )

    def test_standard_errors(self):
        npt.assert_allclose(
            self.oe_r.std_errors.values,
            [self.s["se_int"], self.s["se_w"], self.s["se_x"]],
            rtol=1e-6,
        )


class TestIVRobustOveridentified:
    """HC1 robust IV-2SLS matches Stata ``ivregress, vce(robust)`` (overidentified)."""

    @pytest.fixture(autouse=True)
    def _run(self, df_iv_panel):
        self.s = S_IV_ROBUST
        self.oe_r = oe.iv("y ~ w | x ~ z1 + z2", data=df_iv_panel, cov_type="robust")

    def test_coefficients(self):
        npt.assert_allclose(
            self.oe_r.coefficients.values,
            [self.s["b_int"], self.s["b_w"], self.s["b_x"]],
            rtol=1e-6,
        )

    def test_standard_errors(self):
        npt.assert_allclose(
            self.oe_r.std_errors.values,
            [self.s["se_int"], self.s["se_w"], self.s["se_x"]],
            rtol=1e-6,
        )


class TestIVClusterOveridentified:
    """Cluster-robust IV-2SLS matches Stata ``ivregress, vce(cluster id)`` (overidentified)."""

    @pytest.fixture(autouse=True)
    def _run(self, df_iv_panel):
        self.s = S_IV_CLUSTER_PANEL
        self.oe_r = oe.iv("y ~ w | x ~ z1 + z2", data=df_iv_panel, cluster="id")

    def test_coefficients(self):
        npt.assert_allclose(
            self.oe_r.coefficients.values,
            [self.s["b_int"], self.s["b_w"], self.s["b_x"]],
            rtol=1e-6,
        )

    def test_standard_errors(self):
        npt.assert_allclose(
            self.oe_r.std_errors.values,
            [self.s["se_int"], self.s["se_w"], self.s["se_x"]],
            rtol=1e-6,
        )


class TestIVFEStata:
    """FE IV-2SLS (pyfixest within-path) matches Stata ``xtivreg, fe``.

    The within transform sweeps the intercept, so coefficients/SEs are the
    two slope terms ``w`` and ``x`` only.  Stata ``vce(robust)`` is implemented
    as **cluster-robust by the entity id** on the demeaned data
    (xtivreg.ado ``within`` program, cluster==2 branch -> ``_regress ...,
    cluster(id)``) -- NOT a heteroskedastic HC estimator.  OE reproduces this
    exactly via the ``fe_robust="xtivreg"`` toggle (default for FE).
    """

    @pytest.fixture(autouse=True)
    def _run(self, df_iv_panel):
        self.s = S_IV_FE
        self.oe_nonrobust = oe.iv(
            "y ~ w | x ~ z1 + z2", data=df_iv_panel, entity="id", cov_type="nonrobust"
        )
        self.oe_robust = oe.iv(
            "y ~ w | x ~ z1 + z2", data=df_iv_panel, entity="id", cov_type="robust"
        )

    def test_nonrobust_coefficients(self):
        npt.assert_allclose(
            [self.oe_nonrobust.coefficients["w"], self.oe_nonrobust.coefficients["x"]],
            [self.s["bw_n"], self.s["bx_n"]],
            rtol=1e-6,
        )

    def test_nonrobust_standard_errors(self):
        npt.assert_allclose(
            [self.oe_nonrobust.std_errors["w"], self.oe_nonrobust.std_errors["x"]],
            [self.s["sew_n"], self.s["sex_n"]],
            rtol=1e-6,
        )

    def test_robust_coefficients(self):
        npt.assert_allclose(
            [self.oe_robust.coefficients["w"], self.oe_robust.coefficients["x"]],
            [self.s["bw_r"], self.s["bx_r"]],
            rtol=1e-6,
        )

    def test_robust_standard_errors(self):
        npt.assert_allclose(
            [self.oe_robust.std_errors["w"], self.oe_robust.std_errors["x"]],
            [self.s["sew_r"], self.s["sex_r"]],
            rtol=1e-6,
        )

