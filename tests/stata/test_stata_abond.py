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
