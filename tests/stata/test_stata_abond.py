"""Stata parity tests for Arellano-Bond GMM (SSC: xtabond2)."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from .stata_runner import read_stata


class TestAbondBasic:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = read_stata("abond_basic")
        self.oe_r = oe.abond("y ~ x + z", data=df_panel,
                             entity="entity", time="time",
                             step="two-step", lags=1)

    def test_coefficients(self):
        # Different GMM implementations → different estimates
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_Ly"], self.s["b_x"], self.s["b_z"]],
                            rtol=1e-4)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_Ly"], self.s["se_x"], self.s["se_z"]],
                            rtol=1e-4)


class TestAbondCollapsedOneStep:
    """Collapsed, one-step, non-robust difference GMM — parity with Stata's
    ``xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z) nolevel small``.

    Ground truth extracted directly from Stata (xtabond2 3.7.2) and treated as
    a fixed reference: e(b)=[-0.11984163, 1.1258209, -0.28974145], e(V) diag =
    [0.06085416, 0.03142457, 0.01086979] (SE = [0.24668636, 0.17726977,
    0.10425827]), e(sig2)=0.19753252, e(N)=90, e(j)=4.
    """

    # Fixed Stata reference values (do not re-derive).
    _B = np.array([-0.11984163, 1.1258209, -0.28974145])
    _SE = np.array([0.24668636, 0.17726977, 0.10425827])

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
