"""Stata parity tests for IV / 2SLS."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from .stata_runner import read_stata


class TestIVBasic:
    @pytest.fixture(autouse=True)
    def _run(self, df_iv):
        self.s = read_stata("iv_basic")
        self.oe_r = oe.iv("y ~ x2 | x ~ z", data=df_iv)

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_int"], self.s["b_x2"], self.s["b_x"]],
                            rtol=1e-7)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_int"], self.s["se_x2"], self.s["se_x"]],
                            rtol=1e-7)

    def test_nobs(self):
        assert self.oe_r.nobs == int(self.s["N"])
