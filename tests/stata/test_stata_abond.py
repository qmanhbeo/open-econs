"""Stata parity tests for Arellano-Bond GMM (SSC: xtabond2)."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from .stata_runner import read_stata


class TestAbondCollapsedOneStep:
    """Collapsed, one-step, non-robust difference GMM — parity with Stata's
    ``xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel small``.

    Ground truth extracted from Stata (xtabond2 3.7.2) at ~1e-7:
    b = [-0.11984163, 1.12582088, -0.28974145]
    se = [0.24668636, 0.17726977, 0.10425827]
    sig2 = 0.19753252, N = 90, n_instruments = 4
    ar1 = -1.09075743, ar2 = 0.53219972
    """

    _B = np.array([-0.11984163, 1.12582088, -0.28974145])
    _SE = np.array([0.24668636, 0.17726977, 0.10425827])
    _AR1 = -1.09075743
    _AR2 = 0.53219972

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="one-step", lags=1,
                             exogenous=["x", "z"], collapse=True,
                             robust=False)

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values, self._B, rtol=1e-4)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values, self._SE, rtol=1e-4)

    def test_dimensions(self):
        assert self.oe_r.n_obs == 90
        assert self.oe_r.n_instruments == 4
        assert self.oe_r.n_entities == 30

    def test_ar1(self):
        npt.assert_allclose(self.oe_r.ar1_stat, self._AR1, rtol=1e-4)

    def test_ar2(self):
        npt.assert_allclose(self.oe_r.ar2_stat, self._AR2, rtol=1e-4)


class TestAbondCollapsedTwoStep:
    """Collapsed, two-step, non-robust difference GMM.

    GT: b = [-0.11991842, 1.12511697, -0.28992785]
    se = [0.21366874, 0.15188418, 0.09373466]
    sig2 = 0.19752537, N = 90, n_instruments = 4
    ar1 = -1.26155100, ar2 = 0.50080720
    """

    _B = np.array([-0.11991842, 1.12511697, -0.28992785])
    _SE = np.array([0.21366874, 0.15188418, 0.09373466])
    _AR1 = -1.26155100
    _AR2 = 0.50080720

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="two-step", lags=1,
                             exogenous=["x", "z"], collapse=True,
                             robust=False)

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values, self._B, rtol=1e-4)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values, self._SE, rtol=1e-4)

    def test_dimensions(self):
        assert self.oe_r.n_obs == 90
        assert self.oe_r.n_instruments == 4
        assert self.oe_r.n_entities == 30

    def test_ar1(self):
        npt.assert_allclose(self.oe_r.ar1_stat, self._AR1, rtol=1e-4)

    def test_ar2(self):
        npt.assert_allclose(self.oe_r.ar2_stat, self._AR2, rtol=1e-4)


class TestAbondCollapsedRobust:
    """Collapsed, one-step, robust difference GMM.

    GT: b = [-0.11984163, 1.12582088, -0.28974145]
    se = [0.21366904, 0.15191954, 0.09373868]
    sig2 = 0.19753252, N = 90, n_instruments = 4
    ar1 = -1.26256874, ar2 = 0.50046640
    """

    _B = np.array([-0.11984163, 1.12582088, -0.28974145])
    _SE = np.array([0.21366904, 0.15191954, 0.09373868])
    _AR1 = -1.26256874
    _AR2 = 0.50046640

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="one-step", lags=1,
                             exogenous=["x", "z"], collapse=True,
                             robust=True)

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values, self._B, rtol=1e-4)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values, self._SE, rtol=1e-4)

    def test_dimensions(self):
        assert self.oe_r.n_obs == 90
        assert self.oe_r.n_instruments == 4
        assert self.oe_r.n_entities == 30

    def test_ar1(self):
        npt.assert_allclose(self.oe_r.ar1_stat, self._AR1, rtol=1e-4)

    def test_ar2(self):
        npt.assert_allclose(self.oe_r.ar2_stat, self._AR2, rtol=1e-4)


class TestAbondCollapsedTwoStepRobust:
    """Collapsed, two-step, robust (Windmeijer-corrected) difference GMM.

    GT: b = [-0.11991842, 1.12511697, -0.28992785]
    se = [0.20660985, 0.14580017, 0.09087048]
    sig2 = 0.19752537, N = 90, n_instruments = 4
    ar1 = -1.30019393, ar2 = 0.50196529
    """

    _B = np.array([-0.11991842, 1.12511697, -0.28992785])
    _SE = np.array([0.20660985, 0.14580017, 0.09087048])
    _AR1 = -1.30019393
    _AR2 = 0.50196529

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="two-step", lags=1,
                             exogenous=["x", "z"], collapse=True,
                             robust=True)

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values, self._B, rtol=1e-4)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values, self._SE, rtol=1e-4)

    def test_dimensions(self):
        assert self.oe_r.n_obs == 90
        assert self.oe_r.n_instruments == 4
        assert self.oe_r.n_entities == 30

    def test_ar1(self):
        npt.assert_allclose(self.oe_r.ar1_stat, self._AR1, rtol=1e-4)

    def test_ar2(self):
        npt.assert_allclose(self.oe_r.ar2_stat, self._AR2, rtol=1e-4)
