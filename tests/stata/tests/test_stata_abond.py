"""Stata parity tests for Arellano-Bond GMM (xtabond2 3.7.2).

All 8 flavors (collapsed / non-collapsed × one-step / two-step × non-robust / robust)
verified against live Stata via read_stata().  See tests/stata/generate-fixtures/abond.do.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

import open_econs as oe

from ..stata_runner import read_stata

pytestmark = pytest.mark.stata

RTOL = 1e-6

# Read Stata results once at module level, shared by all test classes.
S = read_stata("abond")


class TestAbondCollapsedOneStep:
    """Collapsed, one-step, non-robust difference GMM."""

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="one-step", lags=1,
                             exogenous=["x", "z"], collapse=True,
                             robust=False)

    def test_dimensions(self):
        assert self.oe_r.n_obs == int(self.s["N_c_1s_nr"])
        assert self.oe_r.n_instruments == int(self.s["j_c_1s_nr"])
        assert self.oe_r.n_entities == int(self.s["N_g_c_1s_nr"])

    def test_coefficients(self):
        expected = [self.s["b_Ly_c_1s_nr"], self.s["b_x_c_1s_nr"], self.s["b_z_c_1s_nr"]]
        npt.assert_allclose(self.oe_r.coefficients.values, expected, rtol=RTOL)

    def test_standard_errors(self):
        expected = [self.s["se_Ly_c_1s_nr"], self.s["se_x_c_1s_nr"], self.s["se_z_c_1s_nr"]]
        npt.assert_allclose(self.oe_r.std_errors.values, expected, rtol=RTOL)

    def test_ar1(self):
        npt.assert_allclose(self.oe_r.ar1_stat, self.s["ar1_c_1s_nr"], rtol=RTOL)

    def test_ar2(self):
        npt.assert_allclose(self.oe_r.ar2_stat, self.s["ar2_c_1s_nr"], rtol=RTOL)


class TestAbondCollapsedTwoStep:
    """Collapsed, two-step, non-robust difference GMM."""

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="two-step", lags=1,
                             exogenous=["x", "z"], collapse=True,
                             robust=False)

    def test_dimensions(self):
        assert self.oe_r.n_obs == int(self.s["N_c_2s_nr"])
        assert self.oe_r.n_instruments == int(self.s["j_c_2s_nr"])
        assert self.oe_r.n_entities == int(self.s["N_g_c_2s_nr"])

    def test_coefficients(self):
        expected = [self.s["b_Ly_c_2s_nr"], self.s["b_x_c_2s_nr"], self.s["b_z_c_2s_nr"]]
        npt.assert_allclose(self.oe_r.coefficients.values, expected, rtol=RTOL)

    def test_standard_errors(self):
        expected = [self.s["se_Ly_c_2s_nr"], self.s["se_x_c_2s_nr"], self.s["se_z_c_2s_nr"]]
        npt.assert_allclose(self.oe_r.std_errors.values, expected, rtol=RTOL)

    def test_ar1(self):
        npt.assert_allclose(self.oe_r.ar1_stat, self.s["ar1_c_2s_nr"], rtol=RTOL)

    def test_ar2(self):
        npt.assert_allclose(self.oe_r.ar2_stat, self.s["ar2_c_2s_nr"], rtol=RTOL)


class TestAbondCollapsedRobust:
    """Collapsed, one-step, robust difference GMM."""

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="one-step", lags=1,
                             exogenous=["x", "z"], collapse=True,
                             robust=True)

    def test_dimensions(self):
        assert self.oe_r.n_obs == int(self.s["N_c_1s_r"])
        assert self.oe_r.n_instruments == int(self.s["j_c_1s_r"])
        assert self.oe_r.n_entities == int(self.s["N_g_c_1s_r"])

    def test_coefficients(self):
        expected = [self.s["b_Ly_c_1s_r"], self.s["b_x_c_1s_r"], self.s["b_z_c_1s_r"]]
        npt.assert_allclose(self.oe_r.coefficients.values, expected, rtol=RTOL)

    def test_standard_errors(self):
        expected = [self.s["se_Ly_c_1s_r"], self.s["se_x_c_1s_r"], self.s["se_z_c_1s_r"]]
        npt.assert_allclose(self.oe_r.std_errors.values, expected, rtol=RTOL)

    def test_ar1(self):
        npt.assert_allclose(self.oe_r.ar1_stat, self.s["ar1_c_1s_r"], rtol=RTOL)

    def test_ar2(self):
        npt.assert_allclose(self.oe_r.ar2_stat, self.s["ar2_c_1s_r"], rtol=RTOL)


class TestAbondCollapsedTwoStepRobust:
    """Collapsed, two-step, robust (Windmeijer-corrected) difference GMM."""

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="two-step", lags=1,
                             exogenous=["x", "z"], collapse=True,
                             robust=True)

    def test_dimensions(self):
        assert self.oe_r.n_obs == int(self.s["N_c_2s_r"])
        assert self.oe_r.n_instruments == int(self.s["j_c_2s_r"])
        assert self.oe_r.n_entities == int(self.s["N_g_c_2s_r"])

    def test_coefficients(self):
        expected = [self.s["b_Ly_c_2s_r"], self.s["b_x_c_2s_r"], self.s["b_z_c_2s_r"]]
        npt.assert_allclose(self.oe_r.coefficients.values, expected, rtol=RTOL)

    def test_standard_errors(self):
        expected = [self.s["se_Ly_c_2s_r"], self.s["se_x_c_2s_r"], self.s["se_z_c_2s_r"]]
        npt.assert_allclose(self.oe_r.std_errors.values, expected, rtol=RTOL)

    def test_ar1(self):
        npt.assert_allclose(self.oe_r.ar1_stat, self.s["ar1_c_2s_r"], rtol=RTOL)

    def test_ar2(self):
        npt.assert_allclose(self.oe_r.ar2_stat, self.s["ar2_c_2s_r"], rtol=RTOL)


class TestAbondNonCollapsedOneStep:
    """Non-collapsed, one-step, non-robust difference GMM."""

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="one-step", lags=1,
                             exogenous=["x", "z"], collapse=False,
                             robust=False)

    def test_dimensions(self):
        assert self.oe_r.n_obs == int(self.s["N_nc_1s_nr"])
        assert self.oe_r.n_instruments == int(self.s["j_nc_1s_nr"])
        assert self.oe_r.n_entities == int(self.s["N_g_nc_1s_nr"])

    def test_coefficients(self):
        expected = [self.s["b_Ly_nc_1s_nr"], self.s["b_x_nc_1s_nr"], self.s["b_z_nc_1s_nr"]]
        npt.assert_allclose(self.oe_r.coefficients.values, expected, rtol=RTOL)

    def test_standard_errors(self):
        expected = [self.s["se_Ly_nc_1s_nr"], self.s["se_x_nc_1s_nr"], self.s["se_z_nc_1s_nr"]]
        npt.assert_allclose(self.oe_r.std_errors.values, expected, rtol=RTOL)

    def test_ar1(self):
        npt.assert_allclose(self.oe_r.ar1_stat, self.s["ar1_nc_1s_nr"], rtol=RTOL)

    def test_ar2(self):
        npt.assert_allclose(self.oe_r.ar2_stat, self.s["ar2_nc_1s_nr"], rtol=RTOL)


class TestAbondNonCollapsedTwoStep:
    """Non-collapsed, two-step, non-robust difference GMM."""

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="two-step", lags=1,
                             exogenous=["x", "z"], collapse=False,
                             robust=False)

    def test_dimensions(self):
        assert self.oe_r.n_obs == int(self.s["N_nc_2s_nr"])
        assert self.oe_r.n_instruments == int(self.s["j_nc_2s_nr"])
        assert self.oe_r.n_entities == int(self.s["N_g_nc_2s_nr"])

    def test_coefficients(self):
        expected = [self.s["b_Ly_nc_2s_nr"], self.s["b_x_nc_2s_nr"], self.s["b_z_nc_2s_nr"]]
        npt.assert_allclose(self.oe_r.coefficients.values, expected, rtol=RTOL)

    def test_standard_errors(self):
        expected = [self.s["se_Ly_nc_2s_nr"], self.s["se_x_nc_2s_nr"], self.s["se_z_nc_2s_nr"]]
        npt.assert_allclose(self.oe_r.std_errors.values, expected, rtol=RTOL)

    def test_ar1(self):
        npt.assert_allclose(self.oe_r.ar1_stat, self.s["ar1_nc_2s_nr"], rtol=RTOL)

    def test_ar2(self):
        npt.assert_allclose(self.oe_r.ar2_stat, self.s["ar2_nc_2s_nr"], rtol=RTOL)


class TestAbondNonCollapsedRobust:
    """Non-collapsed, one-step, robust difference GMM."""

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="one-step", lags=1,
                             exogenous=["x", "z"], collapse=False,
                             robust=True)

    def test_dimensions(self):
        assert self.oe_r.n_obs == int(self.s["N_nc_1s_r"])
        assert self.oe_r.n_instruments == int(self.s["j_nc_1s_r"])
        assert self.oe_r.n_entities == int(self.s["N_g_nc_1s_r"])

    def test_coefficients(self):
        expected = [self.s["b_Ly_nc_1s_r"], self.s["b_x_nc_1s_r"], self.s["b_z_nc_1s_r"]]
        npt.assert_allclose(self.oe_r.coefficients.values, expected, rtol=RTOL)

    def test_standard_errors(self):
        expected = [self.s["se_Ly_nc_1s_r"], self.s["se_x_nc_1s_r"], self.s["se_z_nc_1s_r"]]
        npt.assert_allclose(self.oe_r.std_errors.values, expected, rtol=RTOL)

    def test_ar1(self):
        npt.assert_allclose(self.oe_r.ar1_stat, self.s["ar1_nc_1s_r"], rtol=RTOL)

    def test_ar2(self):
        npt.assert_allclose(self.oe_r.ar2_stat, self.s["ar2_nc_1s_r"], rtol=RTOL)


class TestAbondNonCollapsedTwoStepRobust:
    """Non-collapsed, two-step, robust (Windmeijer-corrected) difference GMM."""

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="two-step", lags=1,
                             exogenous=["x", "z"], collapse=False,
                             robust=True)

    def test_dimensions(self):
        assert self.oe_r.n_obs == int(self.s["N_nc_2s_r"])
        assert self.oe_r.n_instruments == int(self.s["j_nc_2s_r"])
        assert self.oe_r.n_entities == int(self.s["N_g_nc_2s_r"])

    def test_coefficients(self):
        expected = [self.s["b_Ly_nc_2s_r"], self.s["b_x_nc_2s_r"], self.s["b_z_nc_2s_r"]]
        npt.assert_allclose(self.oe_r.coefficients.values, expected, rtol=RTOL)

    def test_standard_errors(self):
        expected = [self.s["se_Ly_nc_2s_r"], self.s["se_x_nc_2s_r"], self.s["se_z_nc_2s_r"]]
        npt.assert_allclose(self.oe_r.std_errors.values, expected, rtol=RTOL)

    def test_ar1(self):
        npt.assert_allclose(self.oe_r.ar1_stat, self.s["ar1_nc_2s_r"], rtol=RTOL)

    def test_ar2(self):
        npt.assert_allclose(self.oe_r.ar2_stat, self.s["ar2_nc_2s_r"], rtol=RTOL)


# ─────────────────────────────────────────────────────────────────────────────
# System GMM (Blundell-Bond) — parity against Stata xtabond2 3.7.2.
# Fixture generated by tests/stata/generate-fixtures/sysgmm.do.  Four flavors:
# one/two-step × non-robust/robust, all collapsed, system=True.
# ─────────────────────────────────────────────────────────────────────────────
S_SYS = read_stata("sysgmm")


class TestSysGMMOneStepNonRobust:
    """System GMM, one-step, non-robust (Blundell-Bond)."""

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S_SYS
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="one-step", lags=1,
                             exogenous=["x", "z"], collapse=True,
                             robust=False, system=True)

    def test_coefficients(self):
        expected = [self.s["b_Ly_c_1s_nr"], self.s["b_x_c_1s_nr"],
                    self.s["b_z_c_1s_nr"], self.s["b_cons_c_1s_nr"]]
        npt.assert_allclose(self.oe_r.coefficients.values, expected, rtol=RTOL)

    def test_ar1(self):
        npt.assert_allclose(self.oe_r.ar1_stat, self.s["ar1_c_1s_nr"], rtol=RTOL)

    def test_ar2(self):
        npt.assert_allclose(self.oe_r.ar2_stat, self.s["ar2_c_1s_nr"], rtol=RTOL)


class TestSysGMMTwoStepNonRobust:
    """System GMM, two-step, non-robust (Blundell-Bond)."""

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S_SYS
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="two-step", lags=1,
                             exogenous=["x", "z"], collapse=True,
                             robust=False, system=True)

    def test_coefficients(self):
        expected = [self.s["b_Ly_c_2s_nr"], self.s["b_x_c_2s_nr"],
                    self.s["b_z_c_2s_nr"], self.s["b_cons_c_2s_nr"]]
        npt.assert_allclose(self.oe_r.coefficients.values, expected, rtol=RTOL)

    def test_ar1(self):
        npt.assert_allclose(self.oe_r.ar1_stat, self.s["ar1_c_2s_nr"], rtol=RTOL)

    def test_ar2(self):
        npt.assert_allclose(self.oe_r.ar2_stat, self.s["ar2_c_2s_nr"], rtol=RTOL)


class TestSysGMMOneStepRobust:
    """System GMM, one-step, robust (Blundell-Bond)."""

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S_SYS
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="one-step", lags=1,
                             exogenous=["x", "z"], collapse=True,
                             robust=True, system=True)

    def test_coefficients(self):
        expected = [self.s["b_Ly_c_1s_r"], self.s["b_x_c_1s_r"],
                    self.s["b_z_c_1s_r"], self.s["b_cons_c_1s_r"]]
        npt.assert_allclose(self.oe_r.coefficients.values, expected, rtol=RTOL)

    def test_ar1(self):
        npt.assert_allclose(self.oe_r.ar1_stat, self.s["ar1_c_1s_r"], rtol=RTOL)

    def test_ar2(self):
        npt.assert_allclose(self.oe_r.ar2_stat, self.s["ar2_c_1s_r"], rtol=RTOL)


class TestSysGMMTwoStepRobust:
    """System GMM, two-step, robust (Windmeijer-corrected, Blundell-Bond)."""

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S_SYS
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="two-step", lags=1,
                             exogenous=["x", "z"], collapse=True,
                             robust=True, system=True)

    def test_coefficients(self):
        expected = [self.s["b_Ly_c_2s_r"], self.s["b_x_c_2s_r"],
                    self.s["b_z_c_2s_r"], self.s["b_cons_c_2s_r"]]
        npt.assert_allclose(self.oe_r.coefficients.values, expected, rtol=RTOL)

    def test_ar1(self):
        npt.assert_allclose(self.oe_r.ar1_stat, self.s["ar1_c_2s_r"], rtol=RTOL)

    def test_ar2(self):
        npt.assert_allclose(self.oe_r.ar2_stat, self.s["ar2_c_2s_r"], rtol=RTOL)
