"""Stata parity tests for IV / 2SLS."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from .stata_runner import run_do, stata_available, DO_DIR

pytestmark = pytest.mark.skipif(
    not stata_available(), reason="StataMP not found"
)


def _stata(label: str) -> dict[str, float]:
    run_do(label)
    df = pd.read_stata(DO_DIR / f"{label}.dta")
    return dict(zip(df["name"], df["value"]))


class TestIVBasic:
    @pytest.fixture(autouse=True)
    def _run(self, df_iv):
        self.s = _stata("iv_basic")
        self.oe_r = oe.iv("y ~ x + x2 | z + x2", data=df_iv)

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_x2"], self.s["b_x"]], rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_x2"], self.s["se_x"]], rtol=1e-6)

    def test_nobs(self):
        assert self.oe_r.nobs == int(self.s["N"])
